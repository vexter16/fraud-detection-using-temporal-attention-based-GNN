#gnn_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import global_mean_pool

class TemporalGNN(torch.nn.Module):
    """
    Graph Neural Network model for fraud detection with temporal attention
    
    This model combines GAT (Graph Attention Networks) with temporal attention 
    to capture both graph structure and time-dependent patterns in transactions.
    """
    def __init__(self, node_features, hidden_channels, num_classes, dropout=0.3):
        super(TemporalGNN, self).__init__()
        
        # Graph convolutional layers
        self.conv1 = GATConv(node_features, hidden_channels, heads=2)
        self.conv2 = GATConv(hidden_channels * 2, hidden_channels, heads=1)
        
        # Temporal attention layer
        self.temporal_attn = nn.Linear(2, 1)  # For edge attributes (amount, time)
        
        # Output layers
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)
        
        # Dropout rate
        self.dropout_rate = dropout
        
    def forward(self, x, edge_index, edge_attr, batch=None):
        """Forward pass through the network"""
        # Apply temporal attention to edge attributes
        edge_weights = torch.sigmoid(self.temporal_attn(edge_attr)).squeeze(-1)
        
        # First Graph Conv layer with edge weights
        x = self.conv1(x, edge_index, edge_weights)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Second Graph Conv layer
        x = self.conv2(x, edge_index, edge_weights)
        x = F.relu(x)
        
        # If we're doing graph classification, pool the nodes
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Final layers
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        
        return x

class TemporalBatchNorm(nn.Module):
    """
    Custom batch normalization for temporal features in GNNs
    
    This normalization helps stabilize training with temporal data by
    normalizing activations across the feature dimension.
    """
    def __init__(self, hidden_channels):
        super(TemporalBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(hidden_channels)
        
    def forward(self, x):
        return self.bn(x)

class EnhancedTemporalGNN(TemporalGNN):
    """
    Enhanced version of TemporalGNN with skip connections and batch normalization
    
    This model builds on the base TemporalGNN by adding:
    1. Skip connections to improve gradient flow
    2. Temporal batch normalization for more stable training
    3. Additional capacity for learning complex temporal patterns
    """
    def __init__(self, node_features, hidden_channels, num_classes, dropout=0.3):
        super().__init__(node_features, hidden_channels, num_classes, dropout)
        # Add temporal batch normalization
        self.temp_norm1 = TemporalBatchNorm(hidden_channels * 2)  # After first GAT layer
        self.temp_norm2 = TemporalBatchNorm(hidden_channels)      # After second GAT layer
        
        # Add skip connections
        self.skip_lin = nn.Linear(node_features, hidden_channels * 2)
        self.skip_lin2 = nn.Linear(hidden_channels * 2, hidden_channels)
        
    def forward(self, x, edge_index, edge_attr, batch=None):
        """Forward pass through the network with skip connections and normalization"""
        # Apply temporal attention to edge attributes
        edge_weights = torch.sigmoid(self.temporal_attn(edge_attr)).squeeze(-1)
        
        # First skip connection preparation
        x_skip = self.skip_lin(x)
        
        # First Graph Conv layer with edge weights
        x = self.conv1(x, edge_index, edge_weights)
        x = self.temp_norm1(x)            # Apply batch normalization
        x = F.relu(x + x_skip)            # Skip connection
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Second skip connection preparation
        x_skip2 = self.skip_lin2(x)
        
        # Second Graph Conv layer
        x = self.conv2(x, edge_index, edge_weights)
        x = self.temp_norm2(x)            # Apply batch normalization
        x = F.relu(x + x_skip2)           # Skip connection
        
        # If we're doing graph classification, pool the nodes
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Final layers
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        
        return x

class GraphSAGEFraudDetector(torch.nn.Module):
    """
    GraphSAGE implementation for fraud detection
    
    This model uses the GraphSAGE algorithm which efficiently generates node embeddings
    by sampling and aggregating features from local neighborhoods.
    """
    def __init__(self, node_features, hidden_channels, num_classes):
        super(GraphSAGEFraudDetector, self).__init__()
        
        # GraphSAGE layers
        self.sage1 = SAGEConv(node_features, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Output layer
        self.lin = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # First SAGE layer
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Second SAGE layer
        x = self.sage2(x, edge_index)
        x = F.relu(x)
        
        # If doing graph classification
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Output layer
        x = self.lin(x)
        
        return x
