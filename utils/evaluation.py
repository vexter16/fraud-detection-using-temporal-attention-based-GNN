#evaluation.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score
)
import seaborn as sns

class FraudEvaluator:
    """
    Evaluation metrics and visualization for fraud detection models
    
    This class provides comprehensive evaluation tools specifically designed
    for fraud detection tasks, including metrics that account for class imbalance.
    """
    
    def __init__(self, model, data):
        """
        Initialize the evaluator with a model and data
        
        Args:
            model: PyTorch model to evaluate
            data: PyG Data object containing the graph
        """
        self.model = model
        self.data = data
        
    def evaluate(self, mask=None):
        """
        Evaluate model performance
        
        Args:
            mask: Optional boolean mask to select specific nodes
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        # Get predictions
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
            pred_probs = torch.softmax(out, dim=1)
            fraud_probs = pred_probs[:, 1]  # Probability of fraud class
            pred_labels = out.argmax(dim=1)
        
        # Apply mask if provided
        if mask is not None:
            y_true = self.data.y[mask]
            y_pred = pred_labels[mask]
            fraud_probs = fraud_probs[mask]
        else:
            y_true = self.data.y
            y_pred = pred_labels
        
        # Convert to numpy for sklearn
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        fraud_probs_np = fraud_probs.cpu().numpy()
        
        # Calculate metrics
        # Fix for single-class case: ensure we always get a 2x2 confusion matrix
        cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'accuracy': accuracy_score(y_true_np, y_pred_np),
            'precision': precision_score(y_true_np, y_pred_np, zero_division=0, labels=[0, 1] if 1 in y_true_np or 1 in y_pred_np else None),
            'recall': recall_score(y_true_np, y_pred_np, zero_division=0, labels=[0, 1] if 1 in y_true_np or 1 in y_pred_np else None),
            'f1': f1_score(y_true_np, y_pred_np, zero_division=0, labels=[0, 1] if 1 in y_true_np or 1 in y_pred_np else None),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
            }
        }
        
        # Calculate ROC curve and AUC
        # Only calculate ROC & PR curves if we have both classes
        if len(np.unique(y_true_np)) > 1 or len(np.unique(y_pred_np)) > 1:
            fpr, tpr, _ = roc_curve(y_true_np, fraud_probs_np)
            metrics['auc'] = auc(fpr, tpr)
            metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
            
            # Calculate Precision-Recall curve and average precision
            precision, recall, _ = precision_recall_curve(y_true_np, fraud_probs_np)
            metrics['pr_auc'] = average_precision_score(y_true_np, fraud_probs_np)
            metrics['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
        else:
            # Default values for single class case
            metrics['auc'] = 0.5
            metrics['roc_curve'] = {'fpr': [0, 1], 'tpr': [0, 1]}
            metrics['pr_auc'] = 0.0
            metrics['pr_curve'] = {'precision': [0, 1], 'recall': [0, 1]}
        
        return metrics
    
    def plot_confusion_matrix(self, mask=None):
        """Plot the confusion matrix"""
        # Evaluate model
        metrics = self.evaluate(mask)
        cm = metrics['confusion_matrix']
        conf_matrix = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud']
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        
        return plt.gcf()
    
    def plot_roc_curve(self, mask=None):
        """Plot the ROC curve"""
        # Evaluate model
        metrics = self.evaluate(mask)
        fpr = np.array(metrics['roc_curve']['fpr'])
        tpr = np.array(metrics['roc_curve']['tpr'])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {metrics["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        
        return plt.gcf()
    
    def plot_precision_recall_curve(self, mask=None):
        """Plot the Precision-Recall curve"""
        # Evaluate model
        metrics = self.evaluate(mask)
        precision = np.array(metrics['pr_curve']['precision'])
        recall = np.array(metrics['pr_curve']['recall'])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                 label=f'PR curve (area = {metrics["pr_auc"]:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        return plt.gcf()
    
    def find_optimal_threshold(self, mask=None, beta=1.0):
        """
        Find optimal classification threshold based on F-beta score
        
        Args:
            mask: Optional mask to select specific nodes
            beta: Beta parameter for F-beta score (beta=1 gives F1)
            
        Returns:
            Optimal threshold value
        """
        self.model.eval()
        
        # Get predictions
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
            fraud_probs = torch.softmax(out, dim=1)[:, 1]  # Probability of fraud
        
        # Apply mask if provided
        if mask is not None:
            y_true = self.data.y[mask]
            fraud_probs = fraud_probs[mask]
        else:
            y_true = self.data.y
        
        # Convert to numpy
        y_true_np = y_true.cpu().numpy()
        fraud_probs_np = fraud_probs.cpu().numpy()
        
        # Try different thresholds
        thresholds = np.linspace(0.1, 0.9, 100)
        f_scores = []
        
        for threshold in thresholds:
            y_pred = (fraud_probs_np >= threshold).astype(int)
            f_score = fbeta_score(y_true_np, y_pred, beta)
            f_scores.append(f_score)
        
        # Find threshold with best F-score
        best_idx = np.argmax(f_scores)
        return thresholds[best_idx]
    
    def get_misclassified_nodes(self, mask=None, threshold=0.5):
        """
        Get nodes that were misclassified by the model
        
        Args:
            mask: Optional mask to select specific nodes
            threshold: Classification threshold
            
        Returns:
            Dictionary with false positives and false negatives
        """
        self.model.eval()
        
        # Get predictions
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
            fraud_probs = torch.softmax(out, dim=1)[:, 1]
        
        # Apply threshold
        pred_labels = (fraud_probs >= threshold).int()
        
        # Apply mask if provided
        if mask is not None:
            indices = torch.arange(self.data.y.size(0))[mask]
            y_true = self.data.y[mask]
            pred_labels = pred_labels[mask]
        else:
            indices = torch.arange(self.data.y.size(0))
            y_true = self.data.y
        
        # Find false positives and false negatives
        fp_mask = (y_true == 0) & (pred_labels == 1)
        fn_mask = (y_true == 1) & (pred_labels == 0)
        
        false_positives = indices[fp_mask].tolist()
        false_negatives = indices[fn_mask].tolist()
        
        return {
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }


def fbeta_score(y_true, y_pred, beta):
    """
    Calculate F-beta score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        beta: Beta parameter
        
    Returns:
        F-beta score
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    if precision == 0 and recall == 0:
        return 0
    
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)


def train_model(model, data, optimizer, epochs=100, val_mask=None, 
               early_stopping=10, verbose=True, class_weights=None):
    """
    Train a GNN model for fraud detection
    
    Args:
        model: The PyTorch model to train
        data: PyG Data object
        optimizer: PyTorch optimizer
        epochs: Maximum number of training epochs
        val_mask: Optional validation mask
        early_stopping: Number of epochs to wait before early stopping
        verbose: Whether to print progress
        class_weights: Optional class weights for handling imbalance
        
    Returns:
        trained model and history dictionary
    """
    # Create train mask if not in data
    if not hasattr(data, 'train_mask'):
        if val_mask is not None:
            # Use everything except validation as training
            train_mask = ~val_mask
        else:
            # Use all data for training
            train_mask = torch.ones(data.y.size(0), dtype=torch.bool)
    else:
        train_mask = data.train_mask
    
    # Default class weights if not provided
    if class_weights is None:
        class_weights = torch.ones(2)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [] if val_mask is not None else None,
        'val_metrics': [] if val_mask is not None else None
    }
    
    # For early stopping
    best_val_loss = float('inf')
    best_model_state = None
    no_improve = 0
    
    # Training loop
    for epoch in range(epochs):
        # Train mode
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index, data.edge_attr)
        
        # Calculate loss with class weights

        loss = torch.nn.functional.cross_entropy(
            out[train_mask], 
            data.y[train_mask], 
            weight=class_weights.to(out.device)
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record training loss
        train_loss = loss.item()
        history['train_loss'].append(train_loss)
        
        # Validation
        if val_mask is not None:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index, data.edge_attr)
                val_loss = torch.nn.functional.cross_entropy(
                    out[val_mask], 
                    data.y[val_mask]
                ).item()
                
                # Calculate validation metrics
                evaluator = FraudEvaluator(model, data)
                val_metrics = evaluator.evaluate(val_mask)
            
            # Record validation metrics
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)
            
            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val F1: {val_metrics['f1']:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_stopping:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
        else:
            # No validation - just print training progress
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
    
    # Load best model if validation was used
    if val_mask is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history
