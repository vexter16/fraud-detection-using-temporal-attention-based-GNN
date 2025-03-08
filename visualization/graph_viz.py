#graph_viz.py
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from torch_geometric.utils import to_networkx

class GraphVisualizer:
    """
    Graph visualization tools for fraud detection networks
    
    This class provides various visualization methods to explore and present
    the transaction graph and fraud detection results.
    """
    
    def __init__(self, graph_data, account_mapping=None):
        """
        Initialize with graph data
        
        Args:
            graph_data: PyG Data object containing the graph
            account_mapping: Optional dictionary mapping node indices to account IDs
        """
        self.graph_data = graph_data
        self.account_mapping = account_mapping
        
        # Convert to networkx for visualization
        self.G = to_networkx(graph_data, to_undirected=False, remove_self_loops=True)
        
        # Add node attributes
        for node in self.G.nodes():
            # Add fraud label
            if graph_data.y is not None:
                self.G.nodes[node]['fraud'] = bool(graph_data.y[node])
            
            # Add account ID if mapping provided
            if account_mapping is not None:
                self.G.nodes[node]['account_id'] = account_mapping.get(node, f"Node_{node}")
    
    def plot_graph(self, figsize=(12, 10), layout='spring', highlight_nodes=None, title=None):
        """
        Plot the transaction graph
        
        Args:
            figsize: Figure size as tuple (width, height)
            layout: Graph layout algorithm ('spring', 'kamada_kawai', etc.)
            highlight_nodes: Optional list of nodes to highlight
            title: Optional plot title
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        plt.figure(figsize=figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.G, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.G)
        elif layout == 'circular':
            pos = nx.circular_layout(self.G)
        else:
            pos = nx.spring_layout(self.G, seed=42)
        
        # Prepare node colors based on fraud label
        fraud_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get('fraud', False)]
        normal_nodes = [n for n, attr in self.G.nodes(data=True) if not attr.get('fraud', False)]
        
        # Draw normal and fraud nodes with different colors
        nx.draw_networkx_nodes(
            self.G, pos, nodelist=normal_nodes, 
            node_color='skyblue', node_size=100, alpha=0.8, label='Normal'
        )
        nx.draw_networkx_nodes(
            self.G, pos, nodelist=fraud_nodes, 
            node_color='red', node_size=150, alpha=0.8, label='Fraud'
        )
        
        # Highlight specific nodes if requested
        if highlight_nodes is not None:
            nx.draw_networkx_nodes(
                self.G, pos, nodelist=highlight_nodes,
                node_color='gold', node_size=200, alpha=1.0, label='Highlighted'
            )
        
        # Draw edges
        nx.draw_networkx_edges(self.G, pos, alpha=0.2, arrowsize=10)
        
        # Add labels if the graph is small enough
        if len(self.G) < 50:
            if self.account_mapping is not None:
                labels = {
                    node: attr.get('account_id', str(node)) 
                    for node, attr in self.G.nodes(data=True)
                }
            else:
                labels = {node: str(node) for node in self.G.nodes()}
                
            nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=8)
        
        # Add title and legend
        if title:
            plt.title(title)
        else:
            plt.title(f"Transaction Graph with {len(fraud_nodes)} Fraud Nodes Identified")
        
        plt.legend()
        plt.axis('off')
        
        return plt.gcf()
    
    def plot_interactive_graph(self, layout='spring', size_by_degree=True):
        """
        Create an interactive plotly visualization of the graph
        
        Args:
            layout: Graph layout algorithm to use
            size_by_degree: Whether to size nodes by their degree
            
        Returns:
            Plotly figure object
        """
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.G, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.G)
        else:
            pos = nx.spring_layout(self.G, seed=42)
        
        # Extract node positions
        x_nodes = [pos[node][0] for node in self.G.nodes()]
        y_nodes = [pos[node][1] for node in self.G.nodes()]
        
        # Prepare node sizes
        if size_by_degree:
            node_sizes = [10 + 5 * self.G.degree(node) for node in self.G.nodes()]
        else:
            node_sizes = [15] * len(self.G.nodes())
        
        # Prepare node colors and hover text
        node_colors = []
        node_text = []
        for node in self.G.nodes():
            # Set color based on fraud label
            if self.G.nodes[node].get('fraud', False):
                node_colors.append('red')
            else:
                node_colors.append('skyblue')
            
            # Create hover text
            if self.account_mapping is not None:
                account_id = self.G.nodes[node].get('account_id', f"Node_{node}")
                text = f"Account: {account_id}<br>Fraud: {self.G.nodes[node].get('fraud', False)}"
            else:
                text = f"Node: {node}<br>Fraud: {self.G.nodes[node].get('fraud', False)}"
            
            node_text.append(text)
        
        # Create edges
        edge_x = []
        edge_y = []
        edge_trace_colors = []
        
        for edge in self.G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Add line (with a bit of curvature to avoid overlap)
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Color edges with known fraud nodes
            if self.G.nodes[edge[0]].get('fraud', False) or self.G.nodes[edge[1]].get('fraud', False):
                edge_trace_colors.extend(['rgba(255,0,0,0.2)'] * 3)
            else:
                edge_trace_colors.extend(['rgba(180,180,180,0.1)'] * 3)
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5),
            hoverinfo='none',
            mode='lines',
            marker=dict(color=edge_trace_colors)
        )
        
        # Create node trace
        node_trace = go.Scatter(
            x=x_nodes, y=y_nodes,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line_width=1,
                line=dict(color='rgb(50,50,50)')
            ),
            text=node_text
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Interactive Transaction Graph',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig
    
    def plot_subgraph(self, center_node, hop_distance=2, figsize=(10, 8)):
        """
        Plot a subgraph centered around a specific node
        
        Args:
            center_node: The central node index
            hop_distance: Number of hops to include in the subgraph
            figsize: Figure size as tuple
            
        Returns:
            Matplotlib figure
        """
        # Extract subgraph
        nodes_to_keep = set([center_node])
        frontier = set([center_node])
        
        # BFS to find nodes within hop_distance
        for _ in range(hop_distance):
            new_frontier = set()
            for node in frontier:
                # Add neighbors
                neighbors = set(self.G.neighbors(node))
                new_frontier.update(neighbors - nodes_to_keep)
                nodes_to_keep.update(neighbors)
            frontier = new_frontier
            if not frontier:
                break
        
        # Create subgraph
        subgraph = self.G.subgraph(nodes_to_keep)
        
        # Plot the subgraph
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(subgraph, seed=42)
        
        # Prepare node colors based on fraud label
        fraud_nodes = [n for n, attr in subgraph.nodes(data=True) if attr.get('fraud', False)]
        normal_nodes = [n for n, attr in subgraph.nodes(data=True) if not attr.get('fraud', False)]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            subgraph, pos, nodelist=[center_node],
            node_color='gold', node_size=300, label='Center Node'
        )
        
        normal_nodes_without_center = [n for n in normal_nodes if n != center_node]
        fraud_nodes_without_center = [n for n in fraud_nodes if n != center_node]
        
        nx.draw_networkx_nodes(
            subgraph, pos, nodelist=normal_nodes_without_center,
            node_color='skyblue', node_size=100, label='Normal'
        )
        nx.draw_networkx_nodes(
            subgraph, pos, nodelist=fraud_nodes_without_center,
            node_color='red', node_size=150, label='Fraud'
        )
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.5, arrows=True)
        
        # Add labels
        if len(subgraph) < 50:
            if self.account_mapping is not None:
                labels = {
                    node: attr.get('account_id', str(node)) 
                    for node, attr in subgraph.nodes(data=True)
                }
            else:
                labels = {node: str(node) for node in subgraph.nodes()}
                
            nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8)
        
        if self.account_mapping is not None:
            center_account = self.account_mapping.get(center_node, f"Node_{center_node}")
            plt.title(f"Subgraph around {center_account} (Node {center_node})")
        else:
            plt.title(f"Subgraph around Node {center_node}")
            
        plt.legend()
        plt.axis('off')
        
        return plt.gcf()
    
    def plot_fraud_distribution(self, figsize=(12, 5)):
        """
        Plot the distribution of fraud vs normal nodes and their connections
        
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize)
        
        # Get fraud and normal nodes
        fraud_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get('fraud', False)]
        normal_nodes = [n for n, attr in self.G.nodes(data=True) if not attr.get('fraud', False)]
        
        # Calculate statistics
        num_fraud = len(fraud_nodes)
        num_normal = len(normal_nodes)
        
        # Connections from normal to fraud
        connections_to_fraud = 0
        edges_within_fraud = 0
        edges_normal_to_normal = 0
        
        for u, v in self.G.edges():
            u_fraud = self.G.nodes[u].get('fraud', False)
            v_fraud = self.G.nodes[v].get('fraud', False)
            
            if u_fraud and v_fraud:
                edges_within_fraud += 1
            elif not u_fraud and not v_fraud:
                edges_normal_to_normal += 1
            elif not u_fraud and v_fraud:
                connections_to_fraud += 1
            # We don't count fraud to normal separately
        
        # Create subplot for node distribution
        plt.subplot(1, 2, 1)
        plt.bar(['Normal', 'Fraud'], [num_normal, num_fraud], color=['skyblue', 'red'])
        plt.title('Distribution of Nodes')
        plt.ylabel('Number of Accounts')
        
        # Create subplot for edge distribution
        plt.subplot(1, 2, 2)
        plt.bar(['Normal→Normal', 'Normal→Fraud', 'Fraud→Fraud'], 
                [edges_normal_to_normal, connections_to_fraud, edges_within_fraud],
                color=['skyblue', 'orange', 'red'])
        plt.title('Distribution of Transactions')
        plt.ylabel('Number of Transactions')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return plt.gcf()
