#data_preprocessing.py
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_transaction_data(file_path):
    """
    Load transaction data from CSV file
    
    Args:
        file_path (str): Path to CSV file containing transaction data
        
    Returns:
        DataFrame: Cleaned transaction data
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transaction data file not found: {file_path}")
    
    # Load data
    print(f"Loading transaction data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Basic data profiling
    print(f"Loaded {len(df)} transactions")
    print(f"Data columns: {df.columns.tolist()}")
    
    # For PaySim data format
    if all(col in df.columns for col in ['step', 'type', 'amount', 'nameOrig', 'nameDest', 'isFraud']):
        df_cleaned = df[['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                      'newbalanceOrig', 'nameDest', 'oldbalanceDest', 
                      'newbalanceDest', 'isFraud']]
        
        # Print fraud statistics
        fraud_count = df_cleaned['isFraud'].sum()
        print(f"Fraud transactions: {fraud_count} ({fraud_count/len(df_cleaned)*100:.2f}%)")
    else:
        # Generic case - keep all columns
        df_cleaned = df
        print("Custom data format detected. Using all columns.")
    
    return df_cleaned

def create_synthetic_data(file_path, num_records=5000, fraud_ratio=0.05):
    """
    Create synthetic transaction data for testing and demonstration
    
    Args:
        file_path (str): Path to save the synthetic data
        num_records (int): Number of records to generate
        fraud_ratio (float): Ratio of fraudulent transactions
        
    Returns:
        str: Path to the saved file
    """
    np.random.seed(42)
    
    # Generate account IDs
    client_accounts = [f"C{i:010d}" for i in range(100)]
    merchant_accounts = [f"M{i:010d}" for i in range(20)]
    
    # Generate transactions
    transactions = []
    for i in range(num_records):
        # Decide if this transaction is fraudulent
        is_fraud = np.random.random() < fraud_ratio
        
        # Determine transaction type and accounts
        if np.random.random() < 0.7:
            # Client to merchant payment (normal)
            txn_type = 'PAYMENT'
            orig = np.random.choice(client_accounts)
            dest = np.random.choice(merchant_accounts)
        else:
            # Client to client transfer
            txn_type = 'TRANSFER'
            orig = np.random.choice(client_accounts)
            dest = np.random.choice(client_accounts)
            while dest == orig:
                dest = np.random.choice(client_accounts)
        
        # Determine amount - fraudulent patterns
        if is_fraud:
            if np.random.random() < 0.3:
                # Very large amount (unusual)
                amount = np.random.uniform(5000, 10000)
            elif np.random.random() < 0.7:
                # Multiple small transfers (structuring)
                amount = np.random.uniform(10, 100)
            else:
                # Amount just below reporting threshold
                amount = np.random.uniform(9500, 9999)
        else:
            # Normal transaction - exponential distribution
            amount = np.random.exponential(500)
            if amount > 10000:  # Cap at realistic value
                amount = 10000
        
        # Generate balances
        old_balance_org = max(0, np.random.normal(5000, 2000))
        new_balance_orig = max(0, old_balance_org - amount)
        old_balance_dest = max(0, np.random.normal(5000, 2000))
        new_balance_dest = old_balance_dest + amount
        
        # Add transaction time patterns
        if is_fraud:
            # Fraudulent transactions often happen at unusual hours
            hour = np.random.choice([0, 1, 2, 3, 4, 22, 23])
        else:

            # Normal business hours more common for legitimate transactions
            # Complete probability array for 24 hours (0-23)
            hour_probs = [0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.06, 
                          0.08, 0.10, 0.09, 0.07, 0.06, 0.05, 0.04, 0.03, 
                          0.03, 0.04, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

            # Verify sum equals 1.0
            assert abs(sum(hour_probs) - 1.0) < 1e-10, f"Sum is {sum(hour_probs)}"

            # Use fixed probabilities
            hour = np.random.choice(range(24), p=hour_probs)
        # Add to list
        transactions.append({
            'step': i // 10,  # Time step (10 transactions per step)
            'hour': hour,
            'type': txn_type,
            'amount': amount,
            'nameOrig': orig,
            'oldbalanceOrg': old_balance_org,
            'newbalanceOrig': new_balance_orig,
            'nameDest': dest,
            'oldbalanceDest': old_balance_dest,
            'newbalanceDest': new_balance_dest,
            'isFraud': int(is_fraud),
            'isFlaggedFraud': int(is_fraud and amount > 5000)  # Flag large fraudulent amounts
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(transactions)
    df.to_csv(file_path, index=False)
    print(f"Created synthetic dataset with {num_records} transactions ({int(fraud_ratio*100)}% fraudulent)")
    print(f"Data saved to {file_path}")
    return file_path

def process_data_for_gnn(df):
    """
    Process transaction data for GNN model
    
    Args:
        df (DataFrame): Transaction dataframe
        
    Returns:
        tuple: (PyG Data object, account_to_idx mapping)
    """
    # Create a mapping from account IDs to node indices
    accounts = list(set(df['nameOrig'].unique()).union(set(df['nameDest'].unique())))
    account_to_idx = {account: idx for idx, account in enumerate(accounts)}
    idx_to_account = {idx: account for account, idx in account_to_idx.items()}
    
    print(f"Graph will have {len(accounts)} nodes (accounts)")
    
    # Create edges (source -> destination)
    edge_index = torch.tensor([
        [account_to_idx[row['nameOrig']] for _, row in df.iterrows()],
        [account_to_idx[row['nameDest']] for _, row in df.iterrows()]
    ], dtype=torch.long)
    
    print(f"Graph will have {edge_index.shape[1]} edges (transactions)")
    
    # Edge features (amount, time step)
    if 'step' in df.columns:
        edge_attr = torch.tensor([
            [row['amount'], row['step']] for _, row in df.iterrows()
        ], dtype=torch.float)
    else:
        # If no time step, use just amount
        edge_attr = torch.tensor([
            [row['amount'], 0] for _, row in df.iterrows()
        ], dtype=torch.float)
    
    # Placeholder node features (will be replaced with computed features)
    node_features = torch.zeros((len(accounts), 4), dtype=torch.float)
    
    # Labels (1 for nodes involved in fraud)
    y = torch.zeros(len(accounts), dtype=torch.long)
    
    # Mark nodes involved in fraud
    if 'isFraud' in df.columns:
        fraud_transactions = df[df['isFraud'] == 1]
        for _, row in fraud_transactions.iterrows():
            # Mark both source and destination nodes
            y[account_to_idx[row['nameOrig']]] = 1
            y[account_to_idx[row['nameDest']]] = 1
    
    # Create PyG data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # Add node features
    data = add_node_features(data, df, account_to_idx)
    
    # Return data and mappings
    return data, account_to_idx, idx_to_account

def add_node_features(data, df, account_to_idx):
    """
    Add meaningful node features to the graph with enhanced temporal patterns
    
    Args:
        data (PyG Data): Graph data
        df (DataFrame): Transaction data
        account_to_idx (dict): Mapping from account IDs to node indices
        
    Returns:
        PyG Data: Updated graph data
    """
    # Extract accounts from mapping
    accounts = list(account_to_idx.keys())
    num_nodes = len(accounts)
    
    # Add temporal features if 'step' exists
    if 'step' in df.columns:
        # Convert step to hour of day (assuming 24 steps = 1 day)
        df['hour'] = df['step'] % 24
        # Add cyclical time encoding to capture time periodicity
        df['time_sin'] = np.sin(2*np.pi*df['hour']/24)
        df['time_cos'] = np.cos(2*np.pi*df['hour']/24)
    
    # Calculate transaction velocity features
    # Use several time windows to capture different patterns
    if 'step' in df.columns:
        window_sizes = [1, 6, 24]  # Hours
        for window in window_sizes:
            # Count of outgoing transactions in window
            df_grouped = df.groupby('nameOrig')
            df[f'txn_count_out_{window}h'] = df.apply(
                lambda row: len(df_grouped.get_group(row['nameOrig']).loc[
                    (df['step'] >= row['step'] - window) & 
                    (df['step'] < row['step'])
                ]) if row['nameOrig'] in df_grouped.groups else 0, 
                axis=1
            )
            
            # Count of incoming transactions in window
            df_grouped = df.groupby('nameDest')
            df[f'txn_count_in_{window}h'] = df.apply(
                lambda row: len(df_grouped.get_group(row['nameDest']).loc[
                    (df['step'] >= row['step'] - window) & 
                    (df['step'] < row['step'])
                ]) if row['nameDest'] in df_grouped.groups else 0, 
                axis=1
            )
    
    # Initialize feature matrix with more features
    # Original 8 features + temporal features + velocity features
    feature_count = 16 if 'step' in df.columns else 8  # Increased from 14 to 16 to fix index error
    node_features = torch.zeros((num_nodes, feature_count), dtype=torch.float)
    
    # For each account, compute features
    for account in accounts:
        idx = account_to_idx[account]
        
        # Transactions where account is sender
        sent = df[df['nameOrig'] == account]
        # Transactions where account is receiver
        received = df[df['nameDest'] == account]
        
        # Original features (1-8)
        # Feature 1: Transaction frequency as sender
        node_features[idx, 0] = len(sent) / max(1, df['step'].max() if 'step' in df.columns else len(df))
        
        # Feature 2: Transaction frequency as receiver
        node_features[idx, 1] = len(received) / max(1, df['step'].max() if 'step' in df.columns else len(df))
        
        # Feature 3: Average sent amount
        node_features[idx, 2] = sent['amount'].mean() if len(sent) > 0 else 0
        
        # Feature 4: Average received amount
        node_features[idx, 3] = received['amount'].mean() if len(received) > 0 else 0
        
        # Feature 5: Variance in sent amounts
        node_features[idx, 4] = sent['amount'].var() if len(sent) > 1 else 0
        
        # Feature 6: Variance in received amounts
        node_features[idx, 5] = received['amount'].var() if len(received) > 1 else 0
        
        # Feature 7: Average balance (if available)
        if 'oldbalanceOrg' in df.columns:
            balance = sent['oldbalanceOrg'].mean() if len(sent) > 0 else 0
            node_features[idx, 6] = balance
        
        # Feature 8: Account type (merchant=1, client=0)
        node_features[idx, 7] = 1.0 if account.startswith('M') else 0.0
        
        # Enhanced features if temporal data available
        if 'step' in df.columns:
            # Feature 9-10: Temporal patterns (cyclical time encoding)
            if len(sent) > 0:
                node_features[idx, 8] = sent['time_sin'].mean()
                node_features[idx, 9] = sent['time_cos'].mean()
            
            # Feature 11-14: Transaction velocity
            for i, window in enumerate(window_sizes):
                # Average outgoing transaction velocity
                if len(sent) > 0:
                    node_features[idx, 10 + i] = sent[f'txn_count_out_{window}h'].mean()
                
                # Average incoming transaction velocity
                if len(received) > 0:
                    node_features[idx, 10 + len(window_sizes) + i] = received[f'txn_count_in_{window}h'].mean()
    
    # Normalize features
    scaler = StandardScaler()
    # Skip last feature (account type) as it's binary
    if num_nodes > 1:  # Only scale if we have enough data
        # Don't normalize binary features
        features_to_normalize = node_features[:, :-1] if feature_count == 8 else node_features[:, :-1]
        node_features[:, :-1] = torch.tensor(
            scaler.fit_transform(features_to_normalize.numpy()),
            dtype=torch.float
        )
    
    # Update node features in the graph
    data.x = node_features
    
    return data

def split_data_for_training(graph_data, test_size=0.2, val_size=0.1):
    """
    Split graph data for training, validation and testing
    
    Args:
        graph_data (PyG Data): Graph data
        test_size (float): Proportion for test set
        val_size (float): Proportion for validation set
        
    Returns:
        tuple: (train_mask, val_mask, test_mask) boolean masks
    """
    # Get total number of nodes
    num_nodes = graph_data.x.size(0)
    
    # Create indices array
    indices = np.arange(num_nodes)
    
    # First split to get test set
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=graph_data.y.numpy()
    )
    
    # Then split the train_val set to get validation set
    train_idx, val_idx = train_test_split(
        train_val_idx, 
        test_size=val_size / (1 - test_size),  # Adjust size to account for previous split
        stratify=graph_data.y.numpy()[train_val_idx]
    )
    
    # Create boolean masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    return train_mask, val_mask, test_mask
