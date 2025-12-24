"""
Neural Navigator - Training Script
Trains the multimodal path prediction model.
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.data_loader import get_dataloaders
from models.model import create_model


class Trainer:
    """Handles model training, validation, and checkpointing."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        vocab: List[str],
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        output_dir: str = 'outputs'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.device = device
        self.output_dir = output_dir
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move data to device
            images = batch['image'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            paths = batch['path'].to(self.device)
            
            # Forward pass
            predictions = self.model(images, tokens)
            loss = self.criterion(predictions, paths)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss
    
    def validate(self) -> float:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                images = batch['image'].to(self.device)
                tokens = batch['tokens'].to(self.device)
                paths = batch['path'].to(self.device)
                
                # Forward pass
                predictions = self.model(images, tokens)
                loss = self.criterion(predictions, paths)
                
                total_loss += loss.item() * images.size(0)
        
        avg_loss = total_loss / len(self.val_loader.dataset)
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'vocab': self.vocab
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'Saved best model with val_loss: {val_loss:.4f}')
    
    def plot_losses(self):
        """Plot and save training curves."""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('Neural Navigator Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f'Saved training curves to {plot_path}')
    
    def train(self, num_epochs: int, save_freq: int = 5):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_freq: Save checkpoint every N epochs
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  ✓ New best validation loss!")
            
            # Save checkpoint
            if epoch % save_freq == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
        
        # Final save and plotting
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        self.save_checkpoint(num_epochs, val_loss, is_best=False)
        self.plot_losses()
        
        # Save training summary
        summary = {
            'num_epochs': num_epochs,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'vocab_size': len(self.vocab),
            'vocab': self.vocab
        }
        
        summary_path = os.path.join(self.output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nTraining summary saved to {summary_path}")


def main(args):
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers
    )
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Vocabulary: {vocab}\n")
    
    # Create model
    print("Creating model...")
    model = create_model(vocab_size=len(vocab), device=device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        device=device,
        learning_rate=args.lr,
        output_dir=output_dir
    )
    
    # Train
    trainer.train(num_epochs=args.epochs, save_freq=args.save_freq)
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Neural Navigator')
    
    # Data arguments
    parser.add_argument('--train_dir', type=str, default='data',
                        help='Path to training data directory')
    parser.add_argument('--test_dir', type=str, default='test_data',
                        help='Path to test data directory')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of training data for validation')
    
    # System arguments
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training (default: use GPU if available)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='neural_navigator/outputs',
                        help='Directory to save outputs')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    main(args)
