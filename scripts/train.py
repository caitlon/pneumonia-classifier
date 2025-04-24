"""Script for training the pneumonia classification model."""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import sys
import io
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, List, Generator, Optional

from pneumonia_classifier.data.dataset import create_dataloaders
from pneumonia_classifier.models.resnet import create_model
from pneumonia_classifier.utils import save_model, get_device, format_metrics
from pneumonia_classifier.config import TRAINING_CONFIG, DATA_DIR, MLFLOW_CONFIG, MODEL_CONFIG


class MLflowLogger:
    """Class for capturing stdout and logging to MLflow."""
    
    def __init__(self):
        """Initialize the logger."""
        self.log_buffer: List[str] = []
        self.original_stdout = sys.stdout
        self.log_stream = io.StringIO()
        self.current_epoch: Optional[int] = None
    
    def write(self, text: str) -> None:
        """
        Write to both the original stdout and the log buffer.
        
        Args:
            text: Text to write
        """
        self.original_stdout.write(text)
        self.log_stream.write(text)
        
        # If we have a complete line, add it to the buffer
        if text.endswith('\n'):
            self.log_buffer.append(self.log_stream.getvalue())
            self.log_stream = io.StringIO()
    
    def flush(self) -> None:
        """Flush the stdout."""
        self.original_stdout.flush()
    
    def log_to_mlflow(self, run_id: str) -> None:
        """
        Log the buffered text to MLflow.
        
        Args:
            run_id: MLflow run ID
        """
        if self.log_buffer:
            log_text = ''.join(self.log_buffer)
            epoch_tag = f"_epoch_{self.current_epoch}" if self.current_epoch is not None else ""
            mlflow.log_text(log_text, f"training_log{epoch_tag}.txt")
            self.log_buffer = []  # Clear buffer after logging
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set the current epoch for logging.
        
        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch


@contextmanager
def capture_logs(run_id: str) -> Generator[MLflowLogger, None, None]:
    """
    Context manager to capture logs and send them to MLflow.
    
    Args:
        run_id: MLflow run ID
        
    Yields:
        Logger instance
    """
    logger = MLflowLogger()
    sys.stdout = logger
    try:
        yield logger
    finally:
        sys.stdout = logger.original_stdout
        logger.log_to_mlflow(run_id)


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, Any],
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    run_name: str = None,
    early_stopping_patience: int = 5
) -> Dict[str, Any]:
    """
    Trains the model and evaluates it on the validation set.
    
    Args:
        model: Model to train
        dataloaders: DataLoaders with training and validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        run_name: Run name for MLflow
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        
    Returns:
        Dict with trained model and metrics
    """
    device = get_device()
    print(f"Using device: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    best_val_loss = float('inf')
    results = {"train_losses": [], "val_losses": [], "val_metrics": []}
    
    # Early stopping variables
    patience_counter = 0
    early_stop = False
    
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_params({
            "model_name": MODEL_CONFIG["model_name"],
            "learning_rate": learning_rate,
            "batch_size": TRAINING_CONFIG["batch_size"],
            "epochs": num_epochs,
            "device": device,
            "pretrained": MODEL_CONFIG["pretrained"],
            "early_stopping_patience": early_stopping_patience
        })
        
        # Log tags
        mlflow.set_tags({
            "task": "classification",
            "domain": "medical",
            "dataset": "pneumonia-xray"
        })
        
        # Setup logging capture
        with capture_logs(run.info.run_id) as logger:
            print(f"Starting training with {num_epochs} epochs")
            print(f"Early stopping patience: {early_stopping_patience}")
            
            for epoch in range(num_epochs):
                logger.set_epoch(epoch)
                
                if early_stop:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
                    
                # Training
                model.train()
                train_loss = 0.0
                
                for inputs, labels in dataloaders["train"]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss = train_loss / len(dataloaders["train"])
                results["train_losses"].append(train_loss)
                
                # Validation
                model.eval()
                val_loss = 0.0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for inputs, labels in dataloaders["val"]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                val_loss = val_loss / len(dataloaders["val"])
                results["val_losses"].append(val_loss)
                
                # Calculate metrics
                metrics = {
                    "accuracy": accuracy_score(all_labels, all_preds),
                    "precision": precision_score(all_labels, all_preds, average='binary'),
                    "recall": recall_score(all_labels, all_preds, average='binary'),
                    "f1": f1_score(all_labels, all_preds, average='binary')
                }
                
                results["val_metrics"].append(metrics)
                
                # Log metrics to MLflow
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"]
                }, step=epoch)
                
                print(f"Epoch {epoch+1}/{num_epochs}, "
                    f"Training loss: {train_loss:.4f}, "
                    f"Validation loss: {val_loss:.4f}, "
                    f"{format_metrics(metrics)}")
                
                # Log current epoch's output
                logger.log_to_mlflow(run.info.run_id)
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    results["best_model"] = model.state_dict()
                    
                    # Save model to MLflow
                    mlflow.pytorch.log_model(model, "best_model")
                    
                    # Reset early stopping counter
                    patience_counter = 0
                    
                    print("New best model saved!")
                else:
                    # Increment early stopping counter
                    patience_counter += 1
                    print(f"No improvement for {patience_counter} epochs")
                    
                    # Check if we should stop early
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered (no improvement for {early_stopping_patience} epochs)")
                        mlflow.set_tag("early_stopped", "True")
                        early_stop = True
            
            # Log final logs
            logger.log_to_mlflow(run.info.run_id)
            
            # Log early stopping information
            if early_stop:
                mlflow.log_param("stopped_epoch", epoch + 1)
            
            # Log best metrics
            best_epoch = results["val_losses"].index(min(results["val_losses"]))
            best_metrics = results["val_metrics"][best_epoch]
            
            mlflow.log_metrics({
                "best_val_loss": results["val_losses"][best_epoch],
                "best_accuracy": best_metrics["accuracy"],
                "best_precision": best_metrics["precision"],
                "best_recall": best_metrics["recall"],
                "best_f1": best_metrics["f1"],
                "best_epoch": best_epoch + 1
            })
            
            # Create a summary file
            summary = (
                f"Training Summary\n"
                f"===============\n\n"
                f"Best epoch: {best_epoch + 1}\n"
                f"Best validation loss: {results['val_losses'][best_epoch]:.4f}\n"
                f"Best accuracy: {best_metrics['accuracy']:.4f}\n"
                f"Best precision: {best_metrics['precision']:.4f}\n"
                f"Best recall: {best_metrics['recall']:.4f}\n"
                f"Best F1-score: {best_metrics['f1']:.4f}\n\n"
                f"Total epochs trained: {epoch + 1}\n"
                f"Early stopped: {early_stop}\n"
            )
            
            mlflow.log_text(summary, "training_summary.txt")
            
            # Remember run ID
            results["run_id"] = run.info.run_id
        
    return results


def main():
    """Main function for training and saving the model."""
    parser = argparse.ArgumentParser(description="Training a pneumonia classification model")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Path to data")
    parser.add_argument("--output_path", type=str, default=TRAINING_CONFIG["default_model_path"], 
                        help="Path to save the model")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["epochs"], 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=TRAINING_CONFIG["learning_rate"], 
                        help="Learning rate")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for MLflow")
    parser.add_argument("--patience", type=int, default=TRAINING_CONFIG["early_stopping_patience"],
                        help="Early stopping patience - number of epochs with no improvement")
    
    args = parser.parse_args()
    
    print("Creating datasets...")
    dataloaders = create_dataloaders(args.data_dir)
    
    print("Creating model...")
    model = create_model()
    
    print("Starting training...")
    results = train_model(
        model=model,
        dataloaders=dataloaders,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        run_name=args.run_name,
        early_stopping_patience=args.patience
    )
    
    print("Saving best model...")
    save_model(results["best_model"], args.output_path)
    
    # Output final metrics
    best_epoch = results["val_losses"].index(min(results["val_losses"]))
    best_metrics = results["val_metrics"][best_epoch]
    
    print("\nBest results:")
    print(f"Epoch: {best_epoch + 1}")
    print(f"Validation loss: {results['val_losses'][best_epoch]:.4f}")
    print(f"{format_metrics(best_metrics)}")
    print(f"MLflow Run ID: {results['run_id']}")
    print(f"To view results run: mlflow ui --port={MLFLOW_CONFIG['ui_port']}")


if __name__ == "__main__":
    main() 