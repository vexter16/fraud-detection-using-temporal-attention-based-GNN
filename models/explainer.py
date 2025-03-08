#explainer.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

class SimpleGNNExplainer:
    """
    A simplified GNN explainer for fraud detection
    
    This class implements techniques to explain why a particular node or transaction
    was flagged as fraudulent by analyzing the graph structure and node influences.
    """
    
    def __init__(self, model, graph_data):
        self.model = model
        self.graph_data = graph_data
        
    def explain_prediction(self, node_idx, num_hops=2):
        """
        Explain why a particular node is flagged as fraudulent
        
        Args:
            node_idx: Index of the node to explain
            num_hops: Number of hops to consider for the subgraph
            
        Returns:
            Dictionary containing explanation details
        """
        # Extract the k-hop subgraph around the target node
        subgraph_nodes, subgraph_edges = self._extract_subgraph(node_idx, num_hops)
        
        # Get predictions for this node
        self.model.eval()
        with torch.no_grad():
            full_pred = self.model(self.graph_data.x, 
                                   self.graph_data.edge_index, 
                                   self.graph_data.edge_attr)
            pred_class = full_pred[node_idx].argmax().item()
            pred_score = torch.softmax(full_pred[node_idx], dim=0)[pred_class].item()
        
        # For each edge in the subgraph, measure its importance
        edge_importance = self._measure_edge_importance(node_idx, subgraph_nodes, subgraph_edges)
        
        # Create a visual explanation
        fig = self._visualize_explanation(node_idx, subgraph_nodes, subgraph_edges, edge_importance, 
                                   pred_class, pred_score)
        
        return {
            'node_idx': node_idx,
            'prediction': pred_class,
            'confidence': pred_score,
            'subgraph_nodes': subgraph_nodes.tolist(),
            'important_edges': [(subgraph_edges[0, i].item(), subgraph_edges[1, i].item()) 
                               for i in range(subgraph_edges.shape[1]) 
                               if edge_importance[i] > 0.5],
            'visualization': fig
        }
    
    def _extract_subgraph(self, node_idx, num_hops):
        """Extract a subgraph around the target node"""
        # Start with just the target node
        subgraph_nodes = set([node_idx])
        
        # For each hop
        frontier = set([node_idx])
        for _ in range(num_hops):
            new_frontier = set()
            # For each node in the current frontier
            for node in frontier:
                # Find all edges connected to this node
                mask1 = (self.graph_data.edge_index[0] == node)
                mask2 = (self.graph_data.edge_index[1] == node)
                connected_nodes = torch.cat([
                    self.graph_data.edge_index[1, mask1],
                    self.graph_data.edge_index[0, mask2]
                ]).unique()
                
                # Add these nodes to the subgraph and new frontier
                subgraph_nodes.update(connected_nodes.tolist())
                new_frontier.update(connected_nodes.tolist())
            
            frontier = new_frontier
        
        # Convert to tensor
        subgraph_nodes = torch.tensor(list(subgraph_nodes), dtype=torch.long)
        
        # Get edges that connect nodes in the subgraph
        mask1 = torch.isin(self.graph_data.edge_index[0], subgraph_nodes)
        mask2 = torch.isin(self.graph_data.edge_index[1], subgraph_nodes)
        mask = mask1 & mask2
        subgraph_edges = self.graph_data.edge_index[:, mask]
        
        return subgraph_nodes, subgraph_edges
    
    def _measure_edge_importance(self, node_idx, subgraph_nodes, subgraph_edges):
        """Measure importance of each edge for the prediction"""
        # Get baseline prediction
        self.model.eval()
        with torch.no_grad():
            full_pred = self.model(self.graph_data.x, 
                                   self.graph_data.edge_index, 
                                   self.graph_data.edge_attr)
            baseline_score = full_pred[node_idx, 1].item()  # Score for class 1 (fraud)
        
        # For each edge, remove it and see how prediction changes
        importance_scores = []
        for i in range(subgraph_edges.shape[1]):
            # Create a version of edges without this edge
            mask = torch.ones(subgraph_edges.shape[1], dtype=torch.bool)
            mask[i] = False
            temp_edges = subgraph_edges[:, mask]
            
            # Get edge attributes for the remaining edges
            edge_mask = torch.zeros(self.graph_data.edge_index.shape[1], dtype=torch.bool)
            for j in range(temp_edges.shape[1]):
                edge = (temp_edges[0, j].item(), temp_edges[1, j].item())
                for k in range(self.graph_data.edge_index.shape[1]):
                    if (self.graph_data.edge_index[0, k].item() == edge[0] and
                        self.graph_data.edge_index[1, k].item() == edge[1]):
                        edge_mask[k] = True
                        break
            
            # If we have edge attributes
            if hasattr(self.graph_data, 'edge_attr') and self.graph_data.edge_attr is not None:
                temp_attr = self.graph_data.edge_attr[edge_mask]
            else:
                temp_attr = None
            
            # Get new prediction
            try:
                with torch.no_grad():
                    pred = self.model(self.graph_data.x, temp_edges, temp_attr)
                    new_score = pred[node_idx, 1].item()
            except:
                # In case of error, assume this edge is not important
                new_score = baseline_score
            
            # Measure importance as change in prediction
            importance = abs(baseline_score - new_score)
            importance_scores.append(importance)
        
        # Normalize importance scores
        if max(importance_scores) > 0:
            importance_scores = [s / max(importance_scores) for s in importance_scores]
        
        return torch.tensor(importance_scores)
    
    def _visualize_explanation(self, node_idx, subgraph_nodes, subgraph_edges, 
                              edge_importance, pred_class, pred_score):
        """Visualize the explanation as a network graph"""
        # Convert to networkx for visualization
        G = nx.DiGraph()
        
        # Add nodes
        for node in subgraph_nodes:
            G.add_node(node.item(), 
                      fraud=(self.graph_data.y[node] == 1),
                      is_target=(node.item() == node_idx))
        
        # Add edges with weights from importance
        for i in range(subgraph_edges.shape[1]):
            src = subgraph_edges[0, i].item()
            dst = subgraph_edges[1, i].item()
            G.add_edge(src, dst, weight=edge_importance[i].item())
        
        # Draw the graph
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        target_nodes = [n for n, attr in G.nodes(data=True) if attr.get('is_target')]
        fraud_nodes = [n for n, attr in G.nodes(data=True) 
                      if attr.get('fraud') and not attr.get('is_target')]
        normal_nodes = [n for n, attr in G.nodes(data=True) 
                       if not attr.get('fraud') and not attr.get('is_target')]
        
        nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_color='red', 
                              node_size=500, label='Target Node')
        nx.draw_networkx_nodes(G, pos, nodelist=fraud_nodes, node_color='orange', 
                              node_size=300, label='Fraud Nodes')
        nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_color='blue', 
                              node_size=100, label='Normal Nodes')
        
        # Draw edges with varying width based on importance
        edges = G.edges()
        weights = [G[u][v]['weight'] * 5 for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7, 
                              edge_color='gray', arrows=True)
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title(f"Node {node_idx} - Prediction: {'Fraud' if pred_class == 1 else 'Normal'} "
                 f"(Confidence: {pred_score:.2f})")
        plt.legend()
        plt.axis('off')
        
        return plt.gcf()  # Return the figure
        
    def generate_text_explanation(self, node_idx, account_id_map=None):
        """Generate a textual explanation for why a node was flagged"""
        explanation = self.explain_prediction(node_idx)
        
        text_parts = []
        
        # Basic prediction information
        pred_type = "fraudulent" if explanation['prediction'] == 1 else "normal"
        text_parts.append(f"Account {account_id_map.get(node_idx, node_idx)} was classified as {pred_type} with "
                          f"{explanation['confidence']*100:.1f}% confidence.")
        
        # Important connections
        if explanation['important_edges']:
            text_parts.append("\nThe following connections influenced this decision:")
            for src, dst in explanation['important_edges'][:5]:  # Show top 5 important edges
                src_name = account_id_map.get(src, src) if account_id_map else src
                dst_name = account_id_map.get(dst, dst) if account_id_map else dst
                text_parts.append(f"- Connection from {src_name} to {dst_name}")
        
        # Behavioral patterns
        text_parts.append("\nBehavioral patterns detected:")
        if explanation['prediction'] == 1:  # If flagged as fraud
            if len(explanation['important_edges']) > 3:
                text_parts.append("- Unusual number of connections to other accounts")
            if explanation['confidence'] > 0.8:
                text_parts.append("- Strong similarity to known fraudulent behavior patterns")
        else:
            text_parts.append("- Transaction patterns consistent with normal behavior")
        
        return "\n".join(text_parts)
