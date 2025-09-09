import logging
from typing import Dict

import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        val_dataloader: DataLoader ,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_path: str = None,
        patience: int = 5,
        scheduler_patience: int =2,
        min_delta: float = 1e-4
) :
    """
    Train the SpeechTextModel

    Args:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        device: Device to train on ('cuda' or 'cpu')
        save_path: Path to save the best model
        patience: Early stopping patience
        min_delta: Minimum improvement for early stopping

    Returns:
        Dictionary containing training history
    """

    self.to(device)

    # Initialize optimizer and loss function
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, self.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=scheduler_patience // 2, factor=0.5
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    logging.info(f"Starting training on {device}")
    logging.info(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    for epoch in range(num_epochs):
        # Training phase
        self.train()
        train_loss = 0.0
        train_predictions = []
        train_labels = []

        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")

        for batch in train_pbar:
            # Move batch to device
            audio = batch['audio'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self(audio, input_ids, attention_mask)
            loss = criterion(outputs['logits'], labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            predictions = torch.argmax(outputs['logits'], dim=1)
            train_predictions.extend(predictions.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            # Update progress bar
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{train_loss / (len(train_predictions) // len(labels)):.4f}"
            })

        # Calculate training metrics
        avg_train_loss = train_loss / len(train_dataloader)
        train_acc = accuracy_score(train_labels, train_predictions)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Validation phase
        val_loss = 0.0
        val_acc = 0.0

        if val_dataloader is not None:
            self.eval()
            val_predictions = []
            val_labels = []

            with torch.no_grad():
                val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation")

                for batch in val_pbar:
                    audio = batch['audio'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = self(audio, input_ids, attention_mask)
                    loss = criterion(outputs['logits'], labels)

                    val_loss += loss.item()
                    predictions = torch.argmax(outputs['logits'], dim=1)
                    val_predictions.extend(predictions.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

                    val_pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{val_loss / (len(val_predictions) // len(labels)):.4f}"
                    })

            avg_val_loss = val_loss / len(val_dataloader)
            val_acc = accuracy_score(val_labels, val_predictions)
            val_f1 = f1_score(val_labels, val_predictions, average='weighted')

            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1

            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
            )

            # Early stopping
            if epochs_without_improvement >= patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        else:
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}"
            )

    logging.info("Training completed!")

    # Save model at the end of training
    if save_path:
        self.save_model(save_path)
        logging.info(f"Model saved at {save_path} after training session")

    return history


def evaluate(
        self,
        dataloader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) :
    """
    Evaluate the model on a dataset

    Args:
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on

    Returns:
        Dictionary containing evaluation metrics
    """
    self.eval()
    self.to(device)

    all_predictions = []
    all_labels = []
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")

        for batch in pbar:
            audio = batch['audio'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = self(audio, input_ids, attention_mask)
            loss = criterion(outputs['logits'], labels)

            total_loss += loss.item()
            predictions = torch.argmax(outputs['logits'], dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': classification_report(all_labels, all_predictions)
    }

    logging.info(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return results


def plot_training_history(self, history: Dict, save_path: str = None):
    """Plot training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plot
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    if 'val_acc' in history and history['val_acc']:
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Learning rate plot
    ax3.plot(epochs, history['learning_rates'], 'g-')
    ax3.set_title('Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True)

    # Combined loss and accuracy
    ax4_twin = ax4.twinx()
    ax4.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax4_twin.plot(epochs, history['train_acc'], 'r-', label='Train Acc')
    if 'val_loss' in history and history['val_loss']:
        ax4.plot(epochs, history['val_loss'], 'b--', label='Val Loss')
    if 'val_acc' in history and history['val_acc']:
        ax4_twin.plot(epochs, history['val_acc'], 'r--', label='Val Acc')

    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss', color='b')
    ax4_twin.set_ylabel('Accuracy', color='r')
    ax4.set_title('Training Progress')

    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Training history plot saved to {save_path}")

    plt.show()
    """
def plot_test_history(self, history: Dict, save_path: str = None):
    fig, ax5 = plt.subplots(figsize=(15, 10))

    # test plot
    epochs = range(1, len(history['test_accuracy']) + 1)
    ax5.plot(epochs, history['test_accuracy'], 'b-', label='Test accuracy')
    if 'test_acc' in history and history['test_acc']:
        ax5.plot(epochs, history['test_acc'], 'b-', label='Test Accuracy')
    ax5.set_title('Test accuracy')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('accuracy')
    ax5.legend()
    ax5.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Test history plot saved to {save_path}")

    plt.show()
"""
