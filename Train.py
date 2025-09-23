import logging
from typing import Dict
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def train_model(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        val_dataloader: DataLoader,
        learning_rate: float = 5e-5,
        weight_decay: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_path: str = None,
        patience: int = 5,
        scheduler_patience: int = 5,
        min_delta: float = 0.001,
        accumulation_steps: int = 4,  # New: gradient accumulation
        warmup_steps: int = 50,      # New: warmup
        label_smoothing: float = 0.0
):
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
        accumulation_steps: Number of steps to accumulate gradients
        warmup_steps: Number of warmup steps for learning rate
        label_smoothing: Label smoothing factor

    Returns:
        Dictionary containing training history
    """

    self.to(device)

    # Initialize optimizer and loss function
    optimizer = optim.AdamW(  # Changed to AdamW for better performance
        filter(lambda p: p.requires_grad, self.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)  # Added label smoothing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=scheduler_patience,
        factor=0.7,  # Fixed: was 07, should be 0.7
        min_lr=1e-7
    )

    # Warmup function - moved inside the function scope
    def get_lr_multiplier(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    # Training history - added missing smoothed loss tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': [],
        'train_loss_smooth': [],  # Added this
        'val_loss_smooth': []     # Added this
    }

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_val_acc = 0.0
    global_step = 0

    logging.info(f"Starting training on {device}")
    logging.info(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
    logging.info(f"Gradient accumulation steps: {accumulation_steps}")
    logging.info(f"Effective batch size: {train_dataloader.batch_size * accumulation_steps}")

    for epoch in range(num_epochs):
        # Training phase
        self.train()
        train_loss = 0.0
        train_predictions = []
        train_labels = []

        batch_losses = []  # Fixed: was batch_losses[]

        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")

        for batch_idx, batch in enumerate(train_pbar):  # Added batch_idx
            # Move batch to device
            audio = batch['audio'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = self(audio, input_ids, attention_mask)
            loss = criterion(outputs['logits'], labels)

            loss = loss / accumulation_steps  # Normalize loss by accumulation steps

            # Backward pass
            loss.backward()
            batch_losses.append(loss.item() * accumulation_steps)

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                # Apply warmup
                lr_mult = get_lr_multiplier(global_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate * lr_mult

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # Track metrics - fixed calculation
            train_loss += loss.item() * accumulation_steps  # Multiply back for correct tracking
            predictions = torch.argmax(outputs['logits'], dim=1)
            train_predictions.extend(predictions.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            # Update progress bar with smoothed loss
            if len(batch_losses) >= 10:
                smooth_loss = np.mean(batch_losses[-10:])
                train_pbar.set_postfix({
                    'loss': f"{loss.item() * accumulation_steps:.4f}",
                    'smooth_loss': f"{smooth_loss:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })

        # Calculate training metrics
        avg_train_loss = train_loss / len(train_dataloader)
        train_acc = accuracy_score(train_labels, train_predictions)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Smoothed loss tracking
        alpha = 0.3
        if len(history['train_loss_smooth']) == 0:
            history['train_loss_smooth'].append(avg_train_loss)
        else:
            smoothed = alpha * avg_train_loss + (1 - alpha) * history['train_loss_smooth'][-1]
            history['train_loss_smooth'].append(smoothed)

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
                        'avg_loss': f"{val_loss / max(1, len(val_predictions) // len(labels)):.4f}"  # Fixed division by zero
                    })

            avg_val_loss = val_loss / len(val_dataloader)
            val_acc = accuracy_score(val_labels, val_predictions)
            val_f1 = f1_score(val_labels, val_predictions, average='weighted')

            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)

            # Add smoothed validation loss
            if len(history['val_loss_smooth']) == 0:
                history['val_loss_smooth'].append(avg_val_loss)
            else:
                smoothed = alpha * avg_val_loss + (1 - alpha) * history['val_loss_smooth'][-1]
                history['val_loss_smooth'].append(smoothed)

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            # Early stopping check - consider using smoothed loss for more stability
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_val_acc = val_acc
                epochs_without_improvement = 0

                # Save best model
                if save_path:
                    best_model_path = save_path.replace('.pth', '_best.pth') if '.pth' in save_path else save_path + '_best.pth'
                    self.save_model(best_model_path)
                    logging.info(f"New best model saved: {best_model_path}")
            else:
                epochs_without_improvement += 1

            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f} (smooth: {history['train_loss_smooth'][-1]:.4f}), "
                f"Train Acc: {train_acc:.4f} - "
                f"Val Loss: {avg_val_loss:.4f} (smooth: {history['val_loss_smooth'][-1]:.4f}), "
                f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
            )

            # Early stopping
            if epochs_without_improvement >= patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                logging.info(f"Best validation accuracy: {best_val_acc:.4f}")
                break

        else:
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}"
            )

    logging.info("Training completed!")

    # Save final model
    if save_path:
        self.save_model(save_path)
        logging.info(f"Final model saved at {save_path}")

    return history


def evaluate(
        self,
        dataloader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
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
            print(outputs['logits'][:10])
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
def plot_test_results(self, results: Dict, save_path: str = None):
    """Plot single test evaluation results (accuracy, f1, loss)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = {k: v for k, v in results.items() if k in ["accuracy", "f1_score", "loss"]}
    ax.bar(metrics.keys(), metrics.values(), color=["blue", "green", "red"])
    ax.set_title("Test Results")
    ax.set_ylabel("Score")
    ax.grid(axis="y")

    for i, (metric, value) in enumerate(metrics.items()):
        ax.text(i, value + 0.01, f"{value:.4f}", ha="center")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Test results plot saved to {save_path}")

    plt.show()

