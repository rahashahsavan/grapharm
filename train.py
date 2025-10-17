#!/usr/bin/env python3
"""
Training script for GraphARM model on ZINC250k dataset.
This script implements the training procedure as described in the GraphARM paper.
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ZINC
import wandb
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from models import GraphARM
from grapharm import GraphARMTrainer
from utils import NodeMasking

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_zinc_dataset(data_dir, subset='train'):
    """
    Load ZINC250k dataset.
    
    Args:
        data_dir: Directory to store the dataset
        subset: 'train', 'val', or 'test'
    
    Returns:
        Dataset object
    """
    dataset = ZINC(root=data_dir, subset=subset, transform=None, pre_transform=None)
    logger.info(f"Loaded {subset} dataset with {len(dataset)} molecules")
    return dataset


def get_model_config():
    """
    Get model configuration based on GraphARM paper.
    
    Returns:
        Dictionary with model configuration
    """
    return {
        'hidden_dim': 256,
        'num_layers': 5,
        'K': 20,  # Number of mixture components
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'M': 4,  # Number of diffusion trajectories per graph
        'max_epochs': 1000,
        'patience': 50,
        'min_lr': 1e-6,
        'lr_decay_factor': 0.5,
        'grad_clip_norm': 1.0
    }


def setup_wandb(config, project_name="GraphARM-ZINC250k"):
    """
    Setup Weights & Biases logging.
    
    Args:
        config: Model configuration
        project_name: W&B project name
    """
wandb.init(
        project=project_name,
        config=config,
        name=f"GraphARM_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=["GraphARM", "ZINC250k", "molecular_generation"]
    )


def train_model(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_zinc_dataset(args.data_dir, 'train')
    val_dataset = load_zinc_dataset(args.data_dir, 'val')
    test_dataset = load_zinc_dataset(args.data_dir, 'test')
    
    # Get model configuration
    config = get_model_config()
    
    # Override config with command line arguments
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.max_epochs:
        config['max_epochs'] = args.max_epochs
    if args.hidden_dim:
        config['hidden_dim'] = args.hidden_dim
    if args.num_layers:
        config['num_layers'] = args.num_layers
    
    # Setup W&B
    if not args.no_wandb:
        setup_wandb(config)
    
    # Get dataset statistics
    num_node_types = train_dataset.x.unique().shape[0]
    num_edge_types = train_dataset.edge_attr.unique().shape[0]
    
    logger.info(f"Number of node types: {num_node_types}")
    logger.info(f"Number of edge types: {num_edge_types}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = GraphARM(
        num_node_types=num_node_types,
        num_edge_types=num_edge_types,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        K=config['K'],
        dropout=config['dropout'],
    device=device
)

    # Initialize trainer
    trainer = GraphARMTrainer(
        model=model,
        dataset=train_dataset,
        device=device,
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        M=config['M']
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_model(args.checkpoint)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['max_epochs']):
        # Training
        trainer.model.train()
        train_losses = []
        val_ordering_losses = []
        
        # Create iterator for validation batches (for ordering network training)
        val_iter = iter(val_loader)
        
        for batch_idx, train_batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            # Convert train batch to list of individual graphs
            train_graphs = [train_batch[i] for i in range(train_batch.num_graphs)]
            
            # Get validation batch for ordering network
            try:
                val_batch = next(val_iter)
            except StopIteration:
                # Restart validation iterator if exhausted
                val_iter = iter(val_loader)
                val_batch = next(val_iter)
            
            val_graphs = [val_batch[i] for i in range(val_batch.num_graphs)]
            
            # Training step: denoising on train_graphs, ordering on val_graphs
            train_loss, val_ordering_loss = trainer.train_step(train_graphs, val_graphs)
            train_losses.append(train_loss)
            if val_ordering_loss is not None:
                val_ordering_losses.append(val_ordering_loss)
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, "
                          f"Denoising Loss: {train_loss:.4f}, "
                          f"Ordering Loss: {val_ordering_loss:.4f if val_ordering_loss else 0:.4f}")
        
        # Validation
        trainer.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                graphs = [batch[i] for i in range(batch.num_graphs)]
                val_loss = trainer.validate(graphs)
                val_losses.append(val_loss)
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_val_ordering_loss = np.mean(val_ordering_losses) if val_ordering_losses else 0.0
        
        logger.info(f"Epoch {epoch+1} - Denoising Loss: {avg_train_loss:.4f}, "
                   f"Ordering Loss: {avg_val_ordering_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            trainer.save_model(f"{args.output_dir}/best_model")
            logger.info(f"New best model saved with val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
        
        # Learning rate scheduling
        if epoch > 0 and epoch % 50 == 0:
            for param_group in trainer.denoising_optimizer.param_groups:
                param_group['lr'] *= config['lr_decay_factor']
            for param_group in trainer.ordering_optimizer.param_groups:
                param_group['lr'] *= config['lr_decay_factor']
            logger.info(f"Reduced learning rate to {trainer.denoising_optimizer.param_groups[0]['lr']:.2e}")
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    test_losses = []
    
    trainer.model.eval()
    with torch.no_grad():
        for batch in test_loader:
            graphs = [batch[i] for i in range(batch.num_graphs)]
            test_loss = trainer.validate(graphs)
            test_losses.append(test_loss)
    
    avg_test_loss = np.mean(test_losses)
    logger.info(f"Final test loss: {avg_test_loss:.4f}")
    
    # Save final model
    trainer.save_model(f"{args.output_dir}/final_model")
    
    # Save training statistics
    stats = {
        'best_val_loss': best_val_loss,
        'final_test_loss': avg_test_loss,
        'total_epochs': epoch + 1,
        'config': config
    }
    
    with open(f"{args.output_dir}/training_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train GraphARM model on ZINC250k')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data/ZINC',
                       help='Directory to store ZINC250k dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Directory to save model checkpoints and outputs')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=5,
                       help='Number of message passing layers')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    
    # Training arguments
    parser.add_argument('--max_epochs', type=int, default=1000,
                       help='Maximum number of epochs')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training')
    
    # Logging arguments
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    train_model(args)


if __name__ == '__main__':
    main()