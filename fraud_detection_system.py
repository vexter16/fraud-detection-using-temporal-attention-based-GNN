#fraud_detection_system.py
import torch
import pandas as pd
import numpy as np
import time
from datetime import datetime
import pickle
import os
import json
from models.gnn_models import TemporalGNN
from models.explainer import SimpleGNNExplainer
from utils.data_processing import process_data_for_gnn, add_node_features
from torch_geometric.data import Data

class FraudDetectionSystem:
    """
    Real-time graph-based fraud detection system
    
    This class provides the core functionality for detecting fraudulent transactions
    using Graph Neural Networks, with support for explanation and visualization.
    """
    
    def __init__(self, model=None, graph_data=None, transaction_df=None, 
                account_to_idx=None, idx_to_account=None, config=None):
        """
        Initialize the fraud detection system
        
        Args:
            model: Optional pre-trained GNN model
            graph_data: Optional PyG Data object representing transaction graph
            transaction_df: Optional DataFrame with transaction history
            account_to_idx: Optional mapping from account IDs to node indices
            idx_to_account: Optional mapping from node indices to account IDs
            config: Optional configuration dictionary
        """
        self.model = model
        self.graph_data = graph_data
        self.transaction_df = transaction_df
        self.account_to_idx = account_to_idx or {}
        self.idx_to_account = idx_to_account or {}
        
        # Default configuration
        self.config = {
            'fraud_threshold': 0.7,
            'suspicious_threshold': 0.5,
            'node_features': 8,
            'hidden_channels': 64,
            'edge_features': 2,
            'explanation_hops': 2
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Initialize explainer if model and data are provided
        if self.model is not None and self.graph_data is not None:
            self.explainer = SimpleGNNExplainer(self.model, self.graph_data)
        else:
            self.explainer = None
        
        # Initialize metrics
        self.metrics = {
            'transactions_processed': 0,
            'fraud_detected': 0,
            'avg_processing_time': 0,
            'detection_history': []
        }
    
    def initialize_model(self, node_features=None, hidden_channels=None, num_classes=2, model_type='temporal_gnn'):
        """
        Initialize a new GNN model
        
        Args:
            node_features: Number of node features (defaults to config value)
            hidden_channels: Number of hidden channels (defaults to config value)
            num_classes: Number of output classes
            model_type: Type of GNN model to use
            
        Returns:
            Initialized model
        """
        if node_features is None:
            node_features = self.config['node_features']
            
        if hidden_channels is None:
            hidden_channels = self.config['hidden_channels']
        
        # Initialize the model based on type
        if model_type == 'enhanced_temporal_gnn':
            from models.gnn_models import EnhancedTemporalGNN
            self.model = EnhancedTemporalGNN(
                node_features=node_features,
                hidden_channels=hidden_channels,
                num_classes=num_classes
            )
        elif model_type == 'graphsage':
            from models.gnn_models import GraphSAGEFraudDetector
            self.model = GraphSAGEFraudDetector(
                node_features=node_features,
                hidden_channels=hidden_channels,
                num_classes=num_classes
            )
        else:  # Default to temporal_gnn
            self.model = TemporalGNN(
                node_features=node_features,
                hidden_channels=hidden_channels,
                num_classes=num_classes
            )
        
        # Initialize explainer if data is available
        if self.graph_data is not None:
            self.explainer = SimpleGNNExplainer(self.model, self.graph_data)
            
        return self.model
    
    def process_transaction(self, transaction):
        """
        Process a single transaction and return fraud score
        
        Args:
            transaction: Dictionary with transaction data
            
        Returns:
            Dictionary with transaction results including fraud score
        """
        start_time = time.time()
        
        # Ensure transaction has required fields
        required_fields = ['nameOrig', 'nameDest', 'amount']
        for field in required_fields:
            if field not in transaction:
                raise ValueError(f"Transaction missing required field: {field}")
        
        # Check if accounts exist in our graph
        if transaction['nameOrig'] not in self.account_to_idx:
            print(f"Warning: Source account {transaction['nameOrig']} not in graph. Adding as new node.")
            self._add_new_account(transaction['nameOrig'])
        
        if transaction['nameDest'] not in self.account_to_idx:
            print(f"Warning: Destination account {transaction['nameDest']} not in graph. Adding as new node.")
            self._add_new_account(transaction['nameDest'])
        
        # Get node indices
        source_idx = self.account_to_idx[transaction['nameOrig']]
        dest_idx = self.account_to_idx[transaction['nameDest']]
        
        # Get fraud score
        fraud_score = self._calculate_fraud_score(source_idx, dest_idx, transaction['amount'])
        
        # Update metrics
        processing_time = time.time() - start_time
        self._update_metrics(fraud_score, processing_time)
        
        result = {
            'transaction': transaction,
            'fraud_score': fraud_score,
            'is_suspicious': fraud_score > self.config['suspicious_threshold'],
            'is_fraud': fraud_score > self.config['fraud_threshold'],
            'processing_time_ms': processing_time * 1000
        }
        
        return result
    
    def process_batch(self, transactions):
        """
        Process a batch of transactions
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List of transaction results
        """
        results = []
        for txn in transactions:
            result = self.process_transaction(txn)
            results.append(result)
        
        return results
    
    def explain_suspicious_transaction(self, transaction_result):
        """
        Generate explanation for why a transaction was flagged as suspicious
        
        Args:
            transaction_result: Result dictionary from process_transaction
            
        Returns:
            Dictionary with explanation
        """
        if not transaction_result['is_suspicious']:
            return {"message": "Transaction was not flagged as suspicious"}
        
        if self.explainer is None:
            return {"message": "Explainer not initialized"}
        
        txn = transaction_result['transaction']
        source_idx = self.account_to_idx[txn['nameOrig']]
        dest_idx = self.account_to_idx[txn['nameDest']]
        
        # Get explanation for source and destination accounts
        source_explanation = self.explainer.explain_prediction(source_idx)
        dest_explanation = self.explainer.explain_prediction(dest_idx)
        
        # Generate text explanation
        text_explanation = self._generate_text_explanation(
            transaction_result, source_explanation, dest_explanation
        )
        
        # Combine explanations
        explanation = {
            'transaction': txn,
            'fraud_score': transaction_result['fraud_score'],
            'source_account': {
                'id': txn['nameOrig'],
                'index': source_idx,
                'suspicious_connections': [
                    self.idx_to_account.get(edge[1], str(edge[1])) 
                    for edge in source_explanation.get('important_edges', [])
                ]
            },
            'destination_account': {
                'id': txn['nameDest'],
                'index': dest_idx,
                'suspicious_connections': [
                    self.idx_to_account.get(edge[1], str(edge[1])) 
                    for edge in dest_explanation.get('important_edges', [])
                ]
            },
            'text_explanation': text_explanation,
            'visualizations': {
                'source': source_explanation.get('visualization'),
                'destination': dest_explanation.get('visualization')
            }
        }
        
        return explanation
    
    def update_graph_with_transaction(self, transaction, fraud_label=None):
        """
        Update the graph with a new transaction
        
        Args:
            transaction: Transaction dictionary
            fraud_label: Optional fraud label (0=normal, 1=fraud, None=unknown)
            
        Returns:
            Updated graph data
        """
        # Ensure accounts exist
        for account in [transaction['nameOrig'], transaction['nameDest']]:
            if account not in self.account_to_idx:
                self._add_new_account(account)
        
        # Get node indices
        src_idx = self.account_to_idx[transaction['nameOrig']]
        dst_idx = self.account_to_idx[transaction['nameDest']]
        
        # Create new edge
        new_edge = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long)
        
        # Create edge attributes
        if 'step' in transaction:
            new_attr = torch.tensor([[transaction['amount'], transaction['step']]], dtype=torch.float)
        else:
            # Use current time step estimation if not provided
            current_step = 0
            if hasattr(self.graph_data, 'edge_attr') and self.graph_data.edge_attr.shape[0] > 0:
                # Estimate next time step based on existing data
                current_step = self.graph_data.edge_attr[:, 1].max().item() + 1
            new_attr = torch.tensor([[transaction['amount'], current_step]], dtype=torch.float)
        
        # Add to graph
        self.graph_data.edge_index = torch.cat([self.graph_data.edge_index, new_edge], dim=1)
        self.graph_data.edge_attr = torch.cat([self.graph_data.edge_attr, new_attr], dim=0)
        
        # Update fraud labels if provided
        if fraud_label is not None:
            # Mark both accounts as fraud or normal
            self.graph_data.y[src_idx] = fraud_label
            self.graph_data.y[dst_idx] = fraud_label
        
        # Update node features
        if self.transaction_df is not None:
            # Add transaction to DataFrame
            new_row = pd.DataFrame([transaction])
            self.transaction_df = pd.concat([self.transaction_df, new_row])
            
            # Recompute features for affected nodes
            for idx in [src_idx, dst_idx]:
                account = self.idx_to_account[idx]
                sent = self.transaction_df[self.transaction_df['nameOrig'] == account]
                received = self.transaction_df[self.transaction_df['nameDest'] == account]
                
                # Update node features based on new transactions
                self._update_node_features(idx, sent, received)
        
        return self.graph_data
    
    def save_model(self, model_path, data_path=None):
        """
        Save the model and optionally the graph data
        
        Args:
            model_path: Path to save the model
            data_path: Optional path to save graph data and mappings
            
        Returns:
            Boolean indicating success
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), model_path)
        
        # Save data if path provided
        if data_path is not None and self.graph_data is not None:
            data_dict = {
                'graph_data': self.graph_data,
                'account_to_idx': self.account_to_idx,
                'idx_to_account': self.idx_to_account,
                'config': self.config,
            }
            with open(data_path, 'wb') as f:
                pickle.dump(data_dict, f)
        
        return True
    
    def load_model(self, model_path, data_path=None):
        """
        Load a saved model and optionally graph data
        
        Args:
            model_path: Path to the saved model
            data_path: Optional path to load graph data and mappings
            
        Returns:
            Boolean indicating success
        """
        # Load graph data first if provided
        if data_path is not None and os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            self.graph_data = data_dict.get('graph_data')
            self.account_to_idx = data_dict.get('account_to_idx', {})
            self.idx_to_account = data_dict.get('idx_to_account', {})
            
            if 'config' in data_dict:
                self.config.update(data_dict['config'])
        
        # Initialize model if needed
        if self.model is None:
            node_features = self.graph_data.x.shape[1] if self.graph_data is not None else self.config['node_features']
            self.model = self.initialize_model(node_features=node_features)
        
        # Load model weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Initialize explainer
        if self.model is not None and self.graph_data is not None:
            self.explainer = SimpleGNNExplainer(self.model, self.graph_data)
        
        return True
    
    def get_metrics(self):
        """
        Get current system metrics
        
        Returns:
            Dictionary of metrics
        """
        return {
            'transactions_processed': self.metrics['transactions_processed'],
            'fraud_detected': self.metrics['fraud_detected'],
            'fraud_rate': self.metrics['fraud_detected'] / max(1, self.metrics['transactions_processed']),
            'avg_processing_time_ms': self.metrics['avg_processing_time'] * 1000,
            'last_10_scores': [h['fraud_score'] for h in self.metrics['detection_history'][-10:]]
        }
    
    def _add_new_account(self, account_id):
        """
        Add a new account to the graph
        
        Args:
            account_id: Account identifier
        """
        if account_id in self.account_to_idx:
            return  # Already exists
            
        # Add to mappings
        new_idx = len(self.account_to_idx)
        self.account_to_idx[account_id] = new_idx
        self.idx_to_account[new_idx] = account_id
        
        if self.graph_data is not None:
            # Expand node features tensor
            new_features = torch.zeros((1, self.graph_data.x.shape[1]), dtype=torch.float)
            self.graph_data.x = torch.cat([self.graph_data.x, new_features], dim=0)
            
            # Expand node labels tensor
            new_label = torch.zeros(1, dtype=torch.long)
            self.graph_data.y = torch.cat([self.graph_data.y, new_label], dim=0)
    
    def _calculate_fraud_score(self, source_idx, dest_idx, amount):
        """
        Calculate fraud score for a transaction
        
        Args:
            source_idx: Source node index
            dest_idx: Destination node index
            amount: Transaction amount
            
        Returns:
            Fraud score between 0 and 1
        """
        if self.model is None:
            # No model - use simple heuristics
            return self._calculate_heuristic_score(source_idx, dest_idx, amount)
        
        # Use model to calculate score
        self.model.eval()
        
        try:
            with torch.no_grad():
                # Get node predictions directly from model's forward pass
                out = self.model(self.graph_data.x, 
                              self.graph_data.edge_index, 
                              self.graph_data.edge_attr)
                
                # Get predictions for source and destination nodes
                source_pred = torch.softmax(out[source_idx], dim=0)
                dest_pred = torch.softmax(out[dest_idx], dim=0)
                
                # Average the probabilities
                combined_pred = (source_pred + dest_pred) / 2
                
                # Get fraud score (probability for class 1)
                fraud_score = combined_pred[1].item()
        except Exception as e:
            print(f"Error calculating fraud score: {str(e)}")
            # Fallback to heuristic
            fraud_score = self._calculate_heuristic_score(source_idx, dest_idx, amount)
        
        return fraud_score
    
    def _calculate_heuristic_score(self, source_idx, dest_idx, amount):
        """Simple heuristic score when model is not available"""
        # High amounts are more suspicious
        amount_factor = min(1.0, amount / 10000)
        
        # Known fraud nodes have higher scores
        source_fraud = self.graph_data.y[source_idx].item() if self.graph_data is not None else 0
        dest_fraud = self.graph_data.y[dest_idx].item() if self.graph_data is not None else 0
        
        # Combine factors
        score = (0.5 * amount_factor) + (0.25 * source_fraud) + (0.25 * dest_fraud)
        
        return score
    
    def _update_metrics(self, fraud_score, processing_time):
        """Update system metrics"""
        self.metrics['transactions_processed'] += 1
        
        if fraud_score > self.config['fraud_threshold']:
            self.metrics['fraud_detected'] += 1
        
        # Update average processing time
        n = self.metrics['transactions_processed']
        prev_avg = self.metrics['avg_processing_time']
        self.metrics['avg_processing_time'] = (prev_avg * (n-1) + processing_time) / n
        
        # Add to history
        self.metrics['detection_history'].append({
            'timestamp': datetime.now().isoformat(),
            'fraud_score': fraud_score,
            'processing_time': processing_time
        })
    
    def _generate_text_explanation(self, transaction_result, source_expl, dest_expl):
        """Generate human-readable explanation text"""
        txn = transaction_result['transaction']
        
        explanation_text = [
            f"Transaction of {txn['amount']} from {txn['nameOrig']} to {txn['nameDest']} "
            f"was flagged with a fraud score of {transaction_result['fraud_score']:.2f}."
        ]
        
        # Add severity level
        if transaction_result['fraud_score'] > self.config['fraud_threshold']:
            explanation_text.append("This transaction is classified as FRAUDULENT.")
        elif transaction_result['fraud_score'] > self.config['suspicious_threshold']:
            explanation_text.append("This transaction is classified as SUSPICIOUS.")
        else:
            explanation_text.append("This transaction appears to be NORMAL.")
        
        # Add source account information
        if source_expl.get('important_edges', []):
            explanation_text.append(f"\nSuspicious patterns for source account {txn['nameOrig']}:")
            for edge in source_expl['important_edges'][:3]:  # Top 3 suspicious connections
                connected_account = self.idx_to_account.get(edge[1], f"Node {edge[1]}")
                explanation_text.append(f"- Suspicious connection to account {connected_account}")
        
        # Add destination account information
        if dest_expl.get('important_edges', []):
            explanation_text.append(f"\nSuspicious patterns for destination account {txn['nameDest']}:")
            for edge in dest_expl['important_edges'][:3]:  # Top 3 suspicious connections
                connected_account = self.idx_to_account.get(edge[1], f"Node {edge[1]}")
                explanation_text.append(f"- Suspicious connection to account {connected_account}")
        
        # Add amount-based insights
        if 'amount' in txn and txn['amount'] > 5000:
            explanation_text.append("\nAmount-based risk factors:")
            explanation_text.append(f"- Large transaction amount (${txn['amount']:.2f})")
        
        # Add temporal patterns if data available
        if self.transaction_df is not None and 'step' in txn:
            # Look for temporal patterns
            recent_txns = self.transaction_df[
                (self.transaction_df['nameOrig'] == txn['nameOrig']) | 
                (self.transaction_df['nameDest'] == txn['nameDest'])
            ]
            
            if len(recent_txns) > 1:
                explanation_text.append("\nRecent activity:")
                recent_txns = recent_txns.sort_values('step', ascending=False).head(3)
                for _, recent in recent_txns.iterrows():
                    explanation_text.append(
                        f"- Step {int(recent['step'])}: {recent['amount']} from "
                        f"{recent['nameOrig']} to {recent['nameDest']}"
                    )
        
        return "\n".join(explanation_text)
    
    def _update_node_features(self, node_idx, sent_txns, received_txns):
        """Update features for a specific node based on transaction history"""
        if self.graph_data is None or sent_txns is None:
            return
        
        # Calculate how many features we need
        has_temporal_features = 'hour' in self.transaction_df.columns or 'step' in self.transaction_df.columns
        feature_count = self.graph_data.x.shape[1]  # Use existing feature count
        
        # Calculate new features
        features = torch.zeros(feature_count, dtype=torch.float)
        
        # Feature 1: Transaction frequency as sender
        features[0] = len(sent_txns) / max(1, self.transaction_df['step'].max() if 'step' in self.transaction_df.columns else len(self.transaction_df))
        
        # Feature 2: Transaction frequency as receiver
        features[1] = len(received_txns) / max(1, self.transaction_df['step'].max() if 'step' in self.transaction_df.columns else len(self.transaction_df))
        
        # Feature 3: Average sent amount
        features[2] = sent_txns['amount'].mean() if len(sent_txns) > 0 else 0
        
        # Feature 4: Average received amount
        features[3] = received_txns['amount'].mean() if len(received_txns) > 0 else 0
        
        # Feature 5: Variance in sent amounts
        features[4] = sent_txns['amount'].var() if len(sent_txns) > 1 else 0
        
        # Feature 6: Variance in received amounts
        features[5] = received_txns['amount'].var() if len(received_txns) > 1 else 0
        
        # Feature 7: Average balance (if available)
        if 'oldbalanceOrg' in sent_txns.columns and len(sent_txns) > 0:
            features[6] = sent_txns['oldbalanceOrg'].mean()
        
        # Feature 8: Account type (merchant=1, client=0)
        account_id = self.idx_to_account[node_idx]
        features[7] = 1.0 if account_id.startswith('M') else 0.0
        
        # Enhanced features if temporal data available
        if has_temporal_features and feature_count > 8:
            # Add cycle time encoding if available
            if 'time_sin' in sent_txns.columns and len(sent_txns) > 0:
                features[8] = sent_txns['time_sin'].mean()
                features[9] = sent_txns['time_cos'].mean()
            
            # Add transaction velocity features if available
            window_sizes = [1, 6, 24]  # Should match those in add_node_features
            for i, window in enumerate(window_sizes):
                col_out = f'txn_count_out_{window}h'
                col_in = f'txn_count_in_{window}h'
                
                if col_out in sent_txns.columns and len(sent_txns) > 0:
                    features[10 + i] = sent_txns[col_out].mean()
                
                if col_in in received_txns.columns and len(received_txns) > 0:
                    features[10 + len(window_sizes) + i] = received_txns[col_in].mean()
        
        # Update the node features
        self.graph_data.x[node_idx] = features
