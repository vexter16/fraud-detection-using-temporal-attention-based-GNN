#main.py
import os
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from models.gnn_models import TemporalGNN, GraphSAGEFraudDetector, EnhancedTemporalGNN
from utils.data_processing import (
    load_transaction_data, 
    create_synthetic_data, 
    process_data_for_gnn, 
    split_data_for_training
)
from utils.evaluation import train_model, FraudEvaluator
from visualization.graph_viz import GraphVisualizer
from fraud_detection_system import FraudDetectionSystem

# Configuration constants
DATA_DIR = "data"
MODEL_DIR = "models/saved"
RESULTS_DIR = "results"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fraud Detection using Graph Neural Networks")
    
    # Main commands
    parser.add_argument('command', type=str, choices=[
        'train', 'evaluate', 'demo', 'generate_data', 'process_transactions'
    ], help='Command to execute')
    
    # Data options
    parser.add_argument('--data_path', type=str, help='Path to transaction data file')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--records', type=int, default=5000, help='Number of synthetic records to generate')
    
    # Model options
    parser.add_argument('--model_type', type=str, default='temporal_gnn', 
                        choices=['temporal_gnn', 'enhanced_temporal_gnn', 'graphsage'], 
                        help='Type of GNN model to use')
    parser.add_argument('--model_path', type=str, help='Path to load/save model')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    
    # Execution options
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    
    return parser.parse_args()


def setup_directories():
    """Create necessary directories"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def get_device(use_cuda=False):
    """Get PyTorch device"""
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def load_or_create_data(args):
    """Load transaction data or create synthetic data"""
    if args.data_path:
        # Use provided data path
        data_path = args.data_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load the data
        df = load_transaction_data(data_path)
    elif args.synthetic:
        # Generate synthetic data
        data_path = os.path.join(DATA_DIR, f"synthetic_data_{args.records}.csv")
        create_synthetic_data(data_path, num_records=args.records)
        df = load_transaction_data(data_path)
    else:
        # No data specified
        raise ValueError("Must specify --data_path or --synthetic")
    
    return df


def prepare_model(args, graph_data=None, device='cpu'):
    """Initialize the fraud detection model"""
    node_features = graph_data.x.shape[1] if graph_data is not None else 8
    
    if args.model_type == 'temporal_gnn':
        model = TemporalGNN(
            node_features=node_features,
            hidden_channels=args.hidden_size,
            num_classes=2
        )
    elif args.model_type == 'enhanced_temporal_gnn':
        # Import directly from models.gnn_models without additional import statement
        model = EnhancedTemporalGNN(
            node_features=node_features,
            hidden_channels=args.hidden_size,
            num_classes=2
        )
    elif args.model_type == 'graphsage':
        model = GraphSAGEFraudDetector(
            node_features=node_features,
            hidden_channels=args.hidden_size,
            num_classes=2
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    return model


def train_command(args):
    """Train a fraud detection model"""
    # Setup device
    device = get_device(args.cuda)
    
    # Load data
    df = load_or_create_data(args)
    
    # Process data for GNN
    graph_data, account_to_idx, idx_to_account = process_data_for_gnn(df)
    graph_data = graph_data.to(device)
    
    # Split data for training
    train_mask, val_mask, test_mask = split_data_for_training(graph_data)
    graph_data.train_mask = train_mask
    graph_data.val_mask = val_mask
    graph_data.test_mask = test_mask
    
    # Create model
    model = prepare_model(args, graph_data, device)
    
    # Handle class imbalance with class weights
    num_normal = (graph_data.y[train_mask] == 0).sum().item()
    num_fraud = (graph_data.y[train_mask] == 1).sum().item()
    
    if num_normal > 0 and num_fraud > 0:
        # Calculate inverse class frequency
        class_weights = torch.tensor([
            1.0 / num_normal,
            1.0 / num_fraud
        ], device=device)
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum()
        class_weights *= 2  # Scale to sum to 2
    else:
        class_weights = torch.tensor([1.0, 1.0], device=device)
    
    print(f"Using class weights: {class_weights}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Train model
    print("Starting training...")
    trained_model, history = train_model(
        model, graph_data, optimizer, 
        epochs=args.epochs, 
        val_mask=val_mask,
        class_weights=class_weights
    )
    
    # Evaluate model
    evaluator = FraudEvaluator(trained_model, graph_data)
    train_metrics = evaluator.evaluate(train_mask)
    val_metrics = evaluator.evaluate(val_mask)
    test_metrics = evaluator.evaluate(test_mask)
    
    print("\nTraining Results:")
    print(f"Train - Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
    print(f"Val   - Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
    print(f"Test  - Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")
    
    # Save model
    if args.model_path:
        model_path = args.model_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_DIR, f"{args.model_type}_{timestamp}.pt")
    
    data_path = model_path.replace('.pt', '_data.pkl')
    
    # Create fraud detection system
    fraud_system = FraudDetectionSystem(
        model=trained_model,
        graph_data=graph_data,
        transaction_df=df,
        account_to_idx=account_to_idx,
        idx_to_account=idx_to_account
    )
    
    # Save the model and data
    fraud_system.save_model(model_path, data_path)
    print(f"Model saved to {model_path}")
    print(f"Data saved to {data_path}")
    
    # Create visualizations if requested
    if args.visualize:
        output_dir = args.output_dir or RESULTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm_plot = evaluator.plot_confusion_matrix(test_mask)
        cm_plot.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        roc_plot = evaluator.plot_roc_curve(test_mask)
        roc_plot.savefig(os.path.join(output_dir, "roc_curve.png"))
        
        # Plot PR curve
        plt.figure(figsize=(10, 8))
        pr_plot = evaluator.plot_precision_recall_curve(test_mask)
        pr_plot.savefig(os.path.join(output_dir, "pr_curve.png"))
        
        # Plot graph visualization
        visualizer = GraphVisualizer(graph_data, account_mapping=idx_to_account)
        
        # Get misclassified nodes
        misclassified = evaluator.get_misclassified_nodes(test_mask)
        if misclassified['false_positives']:
            fp_subgraph = visualizer.plot_subgraph(misclassified['false_positives'][0])
            fp_subgraph.savefig(os.path.join(output_dir, "false_positive_example.png"))
        
        if misclassified['false_negatives']:
            fn_subgraph = visualizer.plot_subgraph(misclassified['false_negatives'][0])
            fn_subgraph.savefig(os.path.join(output_dir, "false_negative_example.png"))
        
        # Plot overall graph distribution
        dist_plot = visualizer.plot_fraud_distribution()
        dist_plot.savefig(os.path.join(output_dir, "fraud_distribution.png"))
        
        print(f"Visualizations saved to {output_dir}")


def evaluate_command(args):
    """Evaluate a trained fraud detection model"""
    # Setup device
    device = get_device(args.cuda)
    
    # Check if model path is provided
    if not args.model_path:
        raise ValueError("Must provide --model_path for evaluation")
    
    # Load fraud detection system
    data_path = args.model_path.replace('.pt', '_data.pkl')
    fraud_system = FraudDetectionSystem()
    fraud_system.load_model(args.model_path, data_path)
    
    # Move to device
    fraud_system.model = fraud_system.model.to(device)
    fraud_system.graph_data = fraud_system.graph_data.to(device)
    
    # Create evaluator
    evaluator = FraudEvaluator(fraud_system.model, fraud_system.graph_data)
    
    # Evaluate overall performance
    metrics = evaluator.evaluate()
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    # Find optimal threshold
    optimal_threshold = evaluator.find_optimal_threshold()
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Create visualizations if requested
    if args.visualize:
        output_dir = args.output_dir or RESULTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot confusion matrix
        cm_plot = evaluator.plot_confusion_matrix()
        cm_plot.savefig(os.path.join(output_dir, "eval_confusion_matrix.png"))
        
        # Plot ROC curve
        roc_plot = evaluator.plot_roc_curve()
        roc_plot.savefig(os.path.join(output_dir, "eval_roc_curve.png"))
        
        # Plot PR curve
        pr_plot = evaluator.plot_precision_recall_curve()
        pr_plot.savefig(os.path.join(output_dir, "eval_pr_curve.png"))
        
        # Plot graph visualization
        visualizer = GraphVisualizer(fraud_system.graph_data, account_mapping=fraud_system.idx_to_account)
        
        # Plot fraud distribution
        dist_plot = visualizer.plot_fraud_distribution()
        dist_plot.savefig(os.path.join(output_dir, "eval_fraud_distribution.png"))
        
        print(f"Visualizations saved to {output_dir}")


def demo_command(args):
    """Run a demonstration of the fraud detection system"""
    # Setup device
    device = get_device(args.cuda)
    
    # Create output directory
    output_dir = args.output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic data if no data path provided
    if not args.data_path:
        data_path = os.path.join(DATA_DIR, "demo_synthetic_data.csv")
        create_synthetic_data(data_path, num_records=args.records, fraud_ratio=0.1)
    else:
        data_path = args.data_path
    
    # Load transaction data
    df = load_transaction_data(data_path)
    
    # Process data for GNN
    graph_data, account_to_idx, idx_to_account = process_data_for_gnn(df)
    graph_data = graph_data.to(device)
    
    # Split data
    train_mask, val_mask, test_mask = split_data_for_training(graph_data)
    graph_data.train_mask = train_mask
    graph_data.val_mask = val_mask
    graph_data.test_mask = test_mask
    
    # Create and train model
    model = prepare_model(args, graph_data, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Training demo model...")
    model, history = train_model(
        model, graph_data, optimizer, 
        epochs=args.epochs, 
        val_mask=val_mask
    )
    
    # Create fraud detection system
    fraud_system = FraudDetectionSystem(
        model=model,
        graph_data=graph_data,
        transaction_df=df,
        account_to_idx=account_to_idx,
        idx_to_account=idx_to_account
    )
    
    # Save the model
    model_path = os.path.join(output_dir, "demo_model.pt")
    data_path = os.path.join(output_dir, "demo_data.pkl")
    fraud_system.save_model(model_path, data_path)
    
    # Run demo transactions
    print("\nProcessing demo transactions...")
    
    # Generate some test transactions
    test_transactions = generate_demo_transactions(fraud_system)
    
    # Process each transaction
    results = []
    for i, txn in enumerate(test_transactions):
        print(f"\nProcessing transaction {i+1}/{len(test_transactions)}")
        result = fraud_system.process_transaction(txn)
        results.append(result)
        
        print(f"  Amount: {txn['amount']:.2f}, From: {txn['nameOrig']}, To: {txn['nameDest']}")
        print(f"  Fraud Score: {result['fraud_score']:.4f}")
        print(f"  Classification: {'Fraud' if result['is_fraud'] else 'Suspicious' if result['is_suspicious'] else 'Normal'}")
        
        # Generate explanation for suspicious transactions
        if result['is_suspicious']:
            explanation = fraud_system.explain_suspicious_transaction(result)
            print("\nExplanation:")
            print(explanation['text_explanation'])
            
            # Save visualization if enabled
            if args.visualize:
                if 'visualizations' in explanation and explanation['visualizations']['source'] is not None:
                    fig_path = os.path.join(output_dir, f"transaction_{i+1}_explanation.png")
                    explanation['visualizations']['source'].savefig(fig_path)
                    plt.close(explanation['visualizations']['source'])
    
    # Print final statistics
    print("\nDemo completed!")
    metrics = fraud_system.get_metrics()
    print(f"Transactions processed: {metrics['transactions_processed']}")
    print(f"Fraud detected: {metrics['fraud_detected']} ({metrics['fraud_rate']*100:.2f}%)")
    print(f"Average processing time: {metrics['avg_processing_time_ms']:.2f} ms")
    
    if args.visualize:
        # Create a visualizer
        visualizer = GraphVisualizer(fraud_system.graph_data, account_mapping=idx_to_account)
        
        # Plot and save overall graph
        graph_plot = visualizer.plot_graph(title="Transaction Network with Fraud Detection")
        graph_plot.savefig(os.path.join(output_dir, "transaction_graph.png"))
        
        # Plot and save subgraph around a fraudulent node
        fraud_nodes = [i for i, f in enumerate(fraud_system.graph_data.y) if f == 1]
        if fraud_nodes:
            subgraph_plot = visualizer.plot_subgraph(fraud_nodes[0], hop_distance=2)
            subgraph_plot.savefig(os.path.join(output_dir, "fraud_subgraph.png"))
        
        print(f"Visualizations saved to {output_dir}")


def generate_data_command(args):
    """Generate synthetic data for fraud detection"""
    # Create output directory
    output_dir = args.output_dir or DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic data
    file_path = os.path.join(output_dir, f"synthetic_data_{args.records}.csv")
    create_synthetic_data(file_path, num_records=args.records)
    
    # Print summary
    df = pd.read_csv(file_path)
    
    print("\nSynthetic data generation summary:")
    print(f"Total transactions: {len(df)}")
    print(f"Fraud transactions: {df['isFraud'].sum()} ({df['isFraud'].mean()*100:.2f}%)")
    print(f"Transaction types: {df['type'].value_counts().to_dict()}")
    print(f"Average transaction amount: ${df['amount'].mean():.2f}")
    print(f"Data saved to: {file_path}")


def process_transactions_command(args):
    """Process transactions using a trained model"""
    # Setup device
    device = get_device(args.cuda)
    
    # Check if model path is provided
    if not args.model_path:
        raise ValueError("Must provide --model_path for processing transactions")
    
    # Check if data path is provided
    if not args.data_path:
        raise ValueError("Must provide --data_path with transactions to process")
    
    # Load fraud detection system
    data_path = args.model_path.replace('.pt', '_data.pkl')
    fraud_system = FraudDetectionSystem()
    fraud_system.load_model(args.model_path, data_path)
    
    # Move to device
    fraud_system.model = fraud_system.model.to(device)
    fraud_system.graph_data = fraud_system.graph_data.to(device)
    
    # Load transactions to process
    transactions_df = pd.read_csv(args.data_path)
    
    print(f"Processing {len(transactions_df)} transactions...")
    
    # Create output directory
    output_dir = args.output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each transaction
    results = []
    for i, row in transactions_df.iterrows():
        # Convert row to dictionary
        txn = row.to_dict()
        
        # Process transaction
        result = fraud_system.process_transaction(txn)
        results.append(result)
        
        # Print progress
        if i % 100 == 0:
            print(f"Processed {i}/{len(transactions_df)} transactions")
    
    # Create results DataFrame
    results_df = pd.DataFrame([
        {
            'transaction_id': i,
            'source': r['transaction']['nameOrig'],
            'destination': r['transaction']['nameDest'],
            'amount': r['transaction']['amount'],
            'fraud_score': r['fraud_score'],
            'is_suspicious': r['is_suspicious'],
            'is_fraud': r['is_fraud'],
            'processing_time_ms': r['processing_time_ms']
        }
        for i, r in enumerate(results)
    ])
    
    # Save results
    results_path = os.path.join(output_dir, "transaction_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Print statistics
    print("\nProcessing completed!")
    print(f"Total transactions: {len(results_df)}")
    print(f"Suspicious transactions: {results_df['is_suspicious'].sum()} ({results_df['is_suspicious'].mean()*100:.2f}%)")
    print(f"Fraudulent transactions: {results_df['is_fraud'].sum()} ({results_df['is_fraud'].mean()*100:.2f}%)")
    print(f"Average processing time: {results_df['processing_time_ms'].mean():.2f} ms")
    print(f"Results saved to: {results_path}")
    
    # Generate explanations for fraudulent transactions if requested
    if args.visualize:
        print("\nGenerating explanations for fraudulent transactions...")
        fraud_results = [r for r in results if r['is_fraud']]
        
        # Limit to 10 explanations to avoid too many files
        for i, result in enumerate(fraud_results[:10]):
            explanation = fraud_system.explain_suspicious_transaction(result)
            
            # Save explanation text
            explanation_path = os.path.join(output_dir, f"fraud_explanation_{i+1}.txt")
            with open(explanation_path, 'w') as f:
                f.write(explanation['text_explanation'])
            
            # Save visualization if available
            if 'visualizations' in explanation and explanation['visualizations']['source'] is not None:
                fig_path = os.path.join(output_dir, f"fraud_visualization_{i+1}.png")
                explanation['visualizations']['source'].savefig(fig_path)
                plt.close(explanation['visualizations']['source'])
        
        print(f"Explanations saved to {output_dir}")


def generate_demo_transactions(fraud_system, num_transactions=10):
    """Generate some demo transactions for testing"""
    # Get existing accounts
    accounts = list(fraud_system.account_to_idx.keys())
    client_accounts = [a for a in accounts if a.startswith('C')]
    merchant_accounts = [a for a in accounts if a.startswith('M')]
    
    # If we don't have enough accounts, create some
    if len(client_accounts) < 5:
        client_accounts = [f"C{i:010d}" for i in range(5)]
    if len(merchant_accounts) < 2:
        merchant_accounts = [f"M{i:010d}" for i in range(2)]
    
    # Generate transactions
    transactions = []
    
    # Normal transaction (client to merchant, medium amount)
    transactions.append({
        'nameOrig': np.random.choice(client_accounts),
        'nameDest': np.random.choice(merchant_accounts),
        'amount': np.random.uniform(100, 500),
        'step': 1,
        'type': 'PAYMENT'
    })
    
    # Very large transaction (potential fraud)
    transactions.append({
        'nameOrig': np.random.choice(client_accounts),
        'nameDest': np.random.choice(merchant_accounts),
        'amount': np.random.uniform(8000, 10000),
        'step': 2,
        'type': 'PAYMENT'
    })
    
    # Series of small transactions from same source (structuring)
    source = np.random.choice(client_accounts)
    dest = np.random.choice(client_accounts)
    while source == dest:
        dest = np.random.choice(client_accounts)
    
    for i in range(3):
        transactions.append({
            'nameOrig': source,
            'nameDest': dest,
            'amount': np.random.uniform(10, 50),
            'step': 3 + i,
            'type': 'TRANSFER'
        })
    
    # Normal client transfer
    source = np.random.choice(client_accounts)
    dest = np.random.choice(client_accounts)
    while source == dest:
        dest = np.random.choice(client_accounts)
    
    transactions.append({
        'nameOrig': source,
        'nameDest': dest,
        'amount': np.random.uniform(200, 1000),
        'step': 6,
        'type': 'TRANSFER'
    })
    
    # Transaction just below reporting threshold (potential structuring)
    transactions.append({
        'nameOrig': np.random.choice(client_accounts),
        'nameDest': np.random.choice(client_accounts),
        'amount': np.random.uniform(9000, 9999),
        'step': 7,
        'type': 'TRANSFER'
    })
    
    # Fill with random transactions to reach num_transactions
    while len(transactions) < num_transactions:
        if np.random.random() < 0.7:
            # Client to merchant
            transactions.append({
                'nameOrig': np.random.choice(client_accounts),
                'nameDest': np.random.choice(merchant_accounts),
                'amount': np.random.exponential(500),
                'step': len(transactions) + 1,
                'type': 'PAYMENT'
            })
        else:
            # Client to client
            source = np.random.choice(client_accounts)
            dest = np.random.choice(client_accounts)
            while source == dest:
                dest = np.random.choice(client_accounts)
                
            transactions.append({
                'nameOrig': source,
                'nameDest': dest,
                'amount': np.random.exponential(500),
                'step': len(transactions) + 1,
                'type': 'TRANSFER'
            })
    
    return transactions


def main():
    """Main entry point"""
    # Set up directories
    setup_directories()
    
    # Parse arguments
    args = parse_args()
    
    # Execute command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'demo':
        demo_command(args)
    elif args.command == 'generate_data':
        generate_data_command(args)
    elif args.command == 'process_transactions':
        process_transactions_command(args)
    else:
        print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
