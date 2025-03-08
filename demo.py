#demo.py
"""
Interactive demonstration script for the fraud detection system

This script provides an interactive demo that showcases the capabilities
of the Graph Neural Network-based fraud detection system.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import torch
import pickle

from models.gnn_models import TemporalGNN
from utils.data_processing import create_synthetic_data, load_transaction_data, process_data_for_gnn
from visualization.graph_viz import GraphVisualizer
from fraud_detection_system import FraudDetectionSystem


def parse_args():
    """Parse command line arguments for the demo"""
    parser = argparse.ArgumentParser(description="Interactive Fraud Detection Demo")
    
    parser.add_argument('--model_path', type=str, help='Path to pre-trained model (optional)')
    parser.add_argument('--data_path', type=str, help='Path to transaction data (optional)')
    parser.add_argument('--records', type=int, default=2000, help='Number of synthetic records if generating data')
    parser.add_argument('--fraud_ratio', type=float, default=0.1, help='Ratio of fraudulent transactions in synthetic data')
    parser.add_argument('--output_dir', type=str, default='demo_results', help='Output directory for results')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--enhanced_model', action='store_true', help='Use enhanced temporal GNN model')
    
    return parser.parse_args()


def get_device(use_cuda=False):
    """Get PyTorch device"""
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def setup_fraud_system(args, device):
    """Set up the fraud detection system"""
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if we're loading a pre-trained model
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading pre-trained model from {args.model_path}")
        
        # Initialize system
        fraud_system = FraudDetectionSystem()
        
        # Load data first to get necessary dimensions
        data_path = args.model_path.replace('.pt', '_data.pkl')
        if os.path.exists(data_path):
            print(f"Loading model data from {data_path}")
            with open(data_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            fraud_system.graph_data = data_dict.get('graph_data')
            fraud_system.account_to_idx = data_dict.get('account_to_idx', {})
            fraud_system.idx_to_account = data_dict.get('idx_to_account', {})
            
            if 'config' in data_dict:
                fraud_system.config.update(data_dict['config'])
        else:
            print(f"Warning: Model data not found at {data_path}")
            
        # Detect model architecture from filename
        model_name = os.path.basename(args.model_path).lower()
        node_features = fraud_system.graph_data.x.shape[1] if fraud_system.graph_data is not None else 8
        
        # Initialize correct model architecture
        if 'graphsage' in model_name:
            print("Detected GraphSAGE model architecture")
            from models.gnn_models import GraphSAGEFraudDetector
            fraud_system.model = GraphSAGEFraudDetector(
                node_features=node_features,
                hidden_channels=64,
                num_classes=2
            )
        else:
            print("Using default TemporalGNN model architecture")
            fraud_system.model = TemporalGNN(
                node_features=node_features,
                hidden_channels=64,
                num_classes=2
            )
        
        # Load model weights
        try:
            # Move to device before loading weights
            fraud_system.model = fraud_system.model.to(device)
            fraud_system.model.load_state_dict(torch.load(args.model_path, map_location=device))
            print("Model weights loaded successfully")
        except Exception as e:
            print(f"Error loading model weights: {str(e)}")
            print("Initializing with random weights instead")
            
        # Move graph data to device
        if fraud_system.graph_data is not None:
            fraud_system.graph_data = fraud_system.graph_data.to(device)
        
    else:
        print("No valid model path provided. Creating and training a new model...")
        
        # Generate or load transaction data
        if args.data_path and os.path.exists(args.data_path):
            print(f"Loading transaction data from {args.data_path}")
            df = load_transaction_data(args.data_path)
        else:
            print(f"Generating synthetic data with {args.records} records")
            data_path = os.path.join('data', f"demo_data_{args.records}.csv")
            create_synthetic_data(data_path, num_records=args.records, fraud_ratio=args.fraud_ratio)
            df = load_transaction_data(data_path)
        
        # Process data for GNN
        print("Converting transactions to graph structure...")
        graph_data, account_to_idx, idx_to_account = process_data_for_gnn(df)
        graph_data = graph_data.to(device)
        
        # Create and train a model
        print("Training a new GNN model...")
        if args.enhanced_model:
            from models.gnn_models import EnhancedTemporalGNN
            model = EnhancedTemporalGNN(
                node_features=graph_data.x.shape[1],
                hidden_channels=64,
                num_classes=2
            ).to(device)
            print("Using enhanced temporal GNN with skip connections and batch normalization")
        else:
            model = TemporalGNN(
                node_features=graph_data.x.shape[1],
                hidden_channels=64,
                num_classes=2
            ).to(device)
            
        # Simple training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        
        # Run 20 epochs for the demo
        for epoch in range(20):
            optimizer.zero_grad()
            out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
            loss = torch.nn.functional.cross_entropy(out, graph_data.y)
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        
        # Create fraud detection system
        fraud_system = FraudDetectionSystem(
            model=model,
            graph_data=graph_data,
            transaction_df=df,
            account_to_idx=account_to_idx,
            idx_to_account=idx_to_account
        )
        
        # Save the model
        model_path = os.path.join(args.output_dir, "demo_model.pt")
        data_path = os.path.join(args.output_dir, "demo_data.pkl")
        fraud_system.save_model(model_path, data_path)
        print(f"Trained model saved to {model_path}")
    
    # Adjust fraud detection thresholds to be more conservative
    print("Adjusting fraud detection thresholds to reduce false positives")
    fraud_system.config['fraud_threshold'] = 0.85  # Increase from 0.7 to 0.85
    fraud_system.config['suspicious_threshold'] = 0.65  # Increase from 0.5 to 0.65
    
    return fraud_system


def print_welcome_message():
    """Print welcome message with ASCII art"""
    welcome_text = """
    ======================================================================
        _____                     _   _____       _            _   _             
       |  ___| __ __ _ _   _  __| | |  __ \  ___| |_ ___  ___| |_(_) ___  _ __  
       | |_ | '__/ _` | | | |/ _` | | |  | |/ _ \ __/ _ \/ __| __| |/ _ \| '_ \ 
       |  _|| | | (_| | |_| | (_| | | |__| |  __/ ||  __/ (__| |_| | (_) | | | |
       |_|  |_|  \__,_|\__,_|\__,_| |_____/ \___|\__\___|\___|\__|_|\___/|_| |_|
                                                                               
         _____ _____  _____      __    _      _               _ _   _           
        / ____|  __ \|  __ \    / _|  | |    | |             (_) | | |          
       | |  __| |__) | |__) |  | |_   | |    | | ___  __ _ _ _| |_| |__  _ __ ___  
       | | |_ |  _  /|  ___/   |  _|  | |    | |/ _ \/ _` | | | __| '_ \| '_ ` _ \ 
       | |__| | | \ \| |       | |    | |____| |  __/ (_| | | | |_| | | | | | | | |
        \_____|_|  \_\_|       |_|    |______|_|\___|\__, |_|_|\__|_| |_|_| |_| |_|
                                                      __/ |                      
                                                     |___/                       
    ======================================================================
    """
    print(welcome_text)
    print("Welcome to the Interactive Fraud Detection System Demo!")
    print("This system uses Graph Neural Networks to detect fraudulent transactions")
    print("in real-time based on the transaction network structure.\n")


def print_menu():
    """Print the interactive menu options"""
    print("\n=== MENU ===")
    print("1. Process a single transaction")
    print("2. Run batch of simulated transactions")
    print("3. View system metrics")
    print("4. Visualize transaction graph")
    print("5. Investigate a suspicious account")
    print("6. View system information")
    print("0. Exit demo")
    print("============")


def process_single_transaction(fraud_system, output_dir):
    """Process a user-defined transaction"""
    print("\n=== Process a Single Transaction ===")
    
    # Get available accounts
    accounts = list(fraud_system.account_to_idx.keys())
    client_accounts = [a for a in accounts if a.startswith('C')][:5]
    merchant_accounts = [a for a in accounts if a.startswith('M')][:3]
    
    print("\nAvailable sender accounts:")
    for i, account in enumerate(client_accounts):
        print(f"{i+1}. {account}")
    
    try:
        sender_idx = int(input("\nSelect sender account number: ")) - 1
        if sender_idx < 0 or sender_idx >= len(client_accounts):
            print("Invalid selection. Using first account.")
            sender_idx = 0
    except ValueError:
        print("Invalid input. Using first account.")
        sender_idx = 0
    
    print("\nAvailable recipient accounts:")
    for i, account in enumerate(merchant_accounts + client_accounts):
        if i < len(merchant_accounts):
            print(f"{i+1}. {account} (Merchant)")
        else:
            # Skip the selected sender account
            client_idx = i - len(merchant_accounts)
            if client_accounts[client_idx] != client_accounts[sender_idx]:
                print(f"{i+1}. {account} (Client)")
    
    try:
        recipient_idx = int(input("\nSelect recipient account number: ")) - 1
        all_recipients = merchant_accounts + client_accounts
        if recipient_idx < 0 or recipient_idx >= len(all_recipients):
            print("Invalid selection. Using first recipient.")
            recipient_idx = 0
    except ValueError:
        print("Invalid input. Using first recipient.")
        recipient_idx = 0
    
    try:
        amount = float(input("\nEnter transaction amount: $"))
    except ValueError:
        print("Invalid amount. Using $100.")
        amount = 100.0
    
    # Create transaction
    transaction = {
        'nameOrig': client_accounts[sender_idx],
        'nameDest': all_recipients[recipient_idx],
        'amount': amount,
        'step': int(time.time()) % 1000,  # Use current time as step
        'type': 'PAYMENT' if recipient_idx < len(merchant_accounts) else 'TRANSFER'
    }
    
    # Process transaction
    print("\nProcessing transaction...")
    start_time = time.time()
    result = fraud_system.process_transaction(transaction)
    processing_time = time.time() - start_time
    
    # Print results
    print("\n=== Transaction Results ===")
    print(f"From: {transaction['nameOrig']}")
    print(f"To: {transaction['nameDest']}")
    print(f"Amount: ${transaction['amount']:.2f}")
    print(f"Type: {transaction['type']}")
    print(f"Fraud Score: {result['fraud_score']:.4f}")
    if result['is_fraud']:
        print("Status: HIGH RISK - FRAUDULENT")
    elif result['is_suspicious']:
        print("Status: MEDIUM RISK - SUSPICIOUS")
    else:
        print("Status: LOW RISK - NORMAL")
    print(f"Processing Time: {processing_time*1000:.2f} ms")
    
    # Generate explanation for suspicious transactions
    if result['is_suspicious'] or result['is_fraud']:
        explanation = fraud_system.explain_suspicious_transaction(result)
        print("\n=== Transaction Explanation ===")
        print(explanation['text_explanation'])
        
        # Save visualization
        if 'visualizations' in explanation and explanation['visualizations']['source'] is not None:
            fig_path = os.path.join(output_dir, "transaction_explanation.png")
            explanation['visualizations']['source'].savefig(fig_path)
            print(f"\nExplanation visualization saved to {fig_path}")
            plt.close(explanation['visualizations']['source'])
    
    return result


def run_batch_simulation(fraud_system, num_transactions=20):
    """Run a batch of simulated transactions"""
    print("\n=== Running Batch Simulation ===")
    print(f"Generating and processing {num_transactions} random transactions...")
    
    # Get available accounts
    accounts = list(fraud_system.account_to_idx.keys())
    client_accounts = [a for a in accounts if a.startswith('C')]
    merchant_accounts = [a for a in accounts if a.startswith('M')]
    
    if not client_accounts or not merchant_accounts:
        print("Not enough accounts available for simulation.")
        return []
    
    # Generate random transactions
    transactions = []
    for _ in range(num_transactions):
        # Random sender (client)
        sender = np.random.choice(client_accounts)
        
        # Random recipient (merchant or client)
        if np.random.random() < 0.7:
            recipient = np.random.choice(merchant_accounts)  # Payment to merchant
            txn_type = 'PAYMENT'
            # Normal payments are usually smaller
            amount = np.random.exponential(500)  # Use exponential distribution for more realistic amounts
            amount = min(amount, 2000)  # Cap at 2000 to avoid too many high-value transactions
        else:
            recipient = np.random.choice(client_accounts)  # Transfer to another client
            while recipient == sender:  # Avoid self-transfers
                recipient = np.random.choice(client_accounts)
            txn_type = 'TRANSFER'
            # Transfers are usually smaller than payments
            amount = np.random.exponential(300)
            amount = min(amount, 1500)
        
        # Create transaction
        transactions.append({
            'nameOrig': sender,
            'nameDest': recipient,
            'amount': amount,
            'step': int(time.time()) % 1000 + _,  # Add index to make steps unique
            'type': txn_type
        })
    
    # Process transactions
    results = []
    for txn in transactions:
        result = fraud_system.process_transaction(txn)
        results.append(result)
    
    # Count results
    num_fraud_detected = sum(1 for r in results if r['is_fraud'])
    num_suspicious_detected = sum(1 for r in results if r['is_suspicious'] and not r['is_fraud'])
    
    print("\n=== Batch Simulation Results ===")
    print(f"Total Transactions: {num_transactions}")
    print(f"Fraudulent Transactions Detected: {num_fraud_detected}")
    print(f"Suspicious Transactions Detected: {num_suspicious_detected}")
    
    return results


def view_system_metrics(fraud_system):
    """View current system metrics"""
    print("\n=== System Metrics ===")
    metrics = fraud_system.get_metrics()
    
    print(f"Transactions Processed: {metrics['transactions_processed']}")
    print(f"Fraud Detected: {metrics['fraud_detected']}")
    print(f"Fraud Rate: {metrics['fraud_rate']*100:.2f}%")
    print(f"Average Processing Time: {metrics['avg_processing_time_ms']:.2f} ms")
    print(f"Last 10 Fraud Scores: {metrics['last_10_scores']}")
    
    return metrics


def visualize_transaction_graph(fraud_system, output_dir):
    """Visualize the transaction graph"""
    print("\n=== Visualize Transaction Graph ===")
    
    visualizer = GraphVisualizer(fraud_system.graph_data, account_mapping=fraud_system.idx_to_account)
    fig = visualizer.plot_graph()
    
    fig_path = os.path.join(output_dir, "transaction_graph.png")
    fig.savefig(fig_path)
    print(f"Transaction graph visualization saved to {fig_path}")
    plt.close(fig)


def investigate_suspicious_account(fraud_system, output_dir):
    """Investigate a suspicious account"""
    print("\n=== Investigate Suspicious Account ===")
    
    # Get available accounts
    accounts = list(fraud_system.account_to_idx.keys())
    client_accounts = [a for a in accounts if a.startswith('C')]
    
    if not client_accounts:
        print("No client accounts found in the system.")
        return
    
    print("\nAvailable accounts:")
    for i, account in enumerate(client_accounts[:10]):
        print(f"{i+1}. {account}")
    
    # Safer input handling
    try:
        account_idx = int(input("\nSelect account number to investigate: ")) - 1
        if account_idx < 0 or account_idx >= len(client_accounts):
            print("Invalid selection. Using first account.")
            account_idx = 0
    except ValueError:
        print("Invalid input. Using first account.")
        account_idx = 0
    
    account_id = client_accounts[account_idx]
    node_idx = fraud_system.account_to_idx[account_id]
    
    # Generate explanation
    if not hasattr(fraud_system, 'explainer') or fraud_system.explainer is None:
        print("Explainer not available. Initializing...")
        from models.explainer import SimpleGNNExplainer
        fraud_system.explainer = SimpleGNNExplainer(fraud_system.model, fraud_system.graph_data)
    
    print("\nGenerating explanation for account using GNN explainer...")
    explanation = fraud_system.explainer.explain_prediction(node_idx)
    
    print("\n=== Account Investigation Results ===")
    print(f"Account ID: {account_id}")
    print(f"Fraud Score: {explanation.get('confidence', 'N/A')}")
    print(f"Prediction: {'Fraudulent' if explanation.get('prediction', 0) == 1 else 'Normal'}")
    
    # Generate textual explanation
    text_explanation = fraud_system.explainer.generate_text_explanation(node_idx, fraud_system.idx_to_account)
    print("\n=== Explanation ===")
    print(text_explanation)
    
    if explanation.get('important_edges', []):
        print("\nSuspicious connections:")
        for edge in explanation.get('important_edges', [])[:5]:  # Show top 5 important edges
            connected_account = fraud_system.idx_to_account.get(edge[1], f"Node {edge[1]}")
            print(f"- Connection to account {connected_account}")
    
    # Save visualization
    if explanation.get('visualization') is not None:
        fig_path = os.path.join(output_dir, "account_investigation.png")
        explanation['visualization'].savefig(fig_path)
        print(f"\nInvestigation visualization saved to {fig_path}")
        plt.close(explanation['visualization'])


def view_system_information(fraud_system):
    """View system information"""
    print("\n=== System Information ===")
    
    print(f"Model Type: {type(fraud_system.model).__name__}")
    print(f"Number of Nodes: {fraud_system.graph_data.num_nodes}")
    print(f"Number of Edges: {fraud_system.graph_data.num_edges}")
    print(f"Node Features: {fraud_system.graph_data.num_node_features}")
    print(f"Edge Features: {fraud_system.graph_data.num_edge_features}")
    
    print("\nConfiguration:")
    for key, value in fraud_system.config.items():
        print(f"{key}: {value}")


def main():
    """Main function for the interactive demo"""
    # Parse command-line arguments
    args = parse_args()
    device = get_device(args.cuda)
    
    # Set up fraud detection system
    fraud_system = setup_fraud_system(args, device)
    
    # Print welcome message
    print_welcome_message()
    
    # Interactive loop
    while True:
        print_menu()
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            process_single_transaction(fraud_system, args.output_dir)
        elif choice == '2':
            run_batch_simulation(fraud_system)
        elif choice == '3':
            view_system_metrics(fraud_system)
        elif choice == '4':
            visualize_transaction_graph(fraud_system, args.output_dir)
        elif choice == '5':
            investigate_suspicious_account(fraud_system, args.output_dir)
        elif choice == '6':
            view_system_information(fraud_system)
        elif choice == '0':
            print("Exiting demo. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()