#!/usr/bin/env python3
"""
Standalone generation script for GraphARM model.
This script loads a trained checkpoint and generates molecules using the autoregressive process.
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import pickle
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC
import json

from models import GraphARM
from utils import NodeMasking

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_checkpoint(checkpoint_path, device):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        Model state dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    else:
        return checkpoint


def load_dataset_info(data_dir):
    """
    Load dataset information to get number of node and edge types.
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        Tuple of (num_node_types, num_edge_types)
    """
    dataset = ZINC(root=data_dir, subset='train', transform=None, pre_transform=None)
    num_node_types = dataset.x.unique().shape[0]
    num_edge_types = dataset.edge_attr.unique().shape[0]
    return num_node_types, num_edge_types


def graph_to_smiles(graph, node_mapping=None, edge_mapping=None):
    """
    Convert a graph to SMILES string.
    This is a simplified implementation - in practice, you'd use RDKit.
    
    Args:
        graph: PyTorch Geometric Data object
        node_mapping: Mapping from node indices to atom types
        edge_mapping: Mapping from edge indices to bond types
        
    Returns:
        SMILES string
    """
    # This is a placeholder implementation
    # In practice, you would use RDKit to convert the graph to SMILES
    n_nodes = graph.x.shape[0]
    n_edges = graph.edge_index.shape[1]
    
    # Create a simple representation
    smiles = f"C{n_nodes}"  # Placeholder: C followed by number of nodes
    
    return smiles


def generate_molecules(model, masker, num_molecules, max_nodes=50, 
                     sampling_method="sample", device='cpu'):
    """
    Generate molecules using the trained model.
    
    Args:
        model: Trained GraphARM model
        masker: NodeMasking object
        num_molecules: Number of molecules to generate
        max_nodes: Maximum number of nodes per molecule
        sampling_method: "sample" or "argmax"
        device: Device to run generation on
        
    Returns:
        List of generated graphs and SMILES strings
    """
    model.eval()
    generated_graphs = []
    generated_smiles = []
    
    logger.info(f"Generating {num_molecules} molecules...")
    
    with torch.no_grad():
        for i in tqdm(range(num_molecules), desc="Generating molecules"):
            try:
                # Start with a single masked node
                current_graph = masker.generate_fully_masked(n_nodes=1)
                current_graph = current_graph.to(device)
                
                # Generate nodes one by one
                for step in range(max_nodes - 1):
                    # Predict next node
                    node_probs, edge_probs = model.denoising_network(current_graph, 0)
                    
                    # Sample node type
                    if sampling_method == "sample":
                        node_type = torch.multinomial(node_probs, 1).item()
                    else:
                        node_type = torch.argmax(node_probs).item()
                    
                    # Sample edge types
                    if sampling_method == "sample":
                        edge_types = torch.multinomial(edge_probs, current_graph.x.shape[0]).squeeze()
                    else:
                        edge_types = torch.argmax(edge_probs, dim=-1)
                    
                    # Demask the current node
                    current_graph = masker.demask_node(current_graph, 0, node_type, edge_types)
                    
                    # Add new masked node
                    current_graph = masker.add_masked_node(current_graph)
                
                # Demask the final node
                node_probs, edge_probs = model.denoising_network(current_graph, 0)
                if sampling_method == "sample":
                    node_type = torch.multinomial(node_probs, 1).item()
                    edge_types = torch.multinomial(edge_probs, current_graph.x.shape[0]).squeeze()
                else:
                    node_type = torch.argmax(node_probs).item()
                    edge_types = torch.argmax(edge_probs, dim=-1)
                
                current_graph = masker.demask_node(current_graph, 0, node_type, edge_types)
                
                # Remove empty edges
                current_graph = masker.remove_empty_edges(current_graph)
                
                # Convert to SMILES
                smiles = graph_to_smiles(current_graph)
                
                generated_graphs.append(current_graph.cpu())
                generated_smiles.append(smiles)
                
            except Exception as e:
                logger.warning(f"Failed to generate molecule {i}: {e}")
                continue
    
    logger.info(f"Successfully generated {len(generated_graphs)} molecules")
    return generated_graphs, generated_smiles


def save_results(graphs, smiles, output_dir):
    """
    Save generated molecules to files.
    
    Args:
        graphs: List of generated graphs
        smiles: List of SMILES strings
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save SMILES strings
    smiles_file = os.path.join(output_dir, "generated_smiles.txt")
    with open(smiles_file, 'w') as f:
        for smi in smiles:
            f.write(f"{smi}\n")
    
    logger.info(f"Saved {len(smiles)} SMILES strings to {smiles_file}")
    
    # Save graph structures
    graphs_file = os.path.join(output_dir, "generated_graphs.pkl")
    with open(graphs_file, 'wb') as f:
        pickle.dump(graphs, f)
    
    logger.info(f"Saved {len(graphs)} graph structures to {graphs_file}")
    
    # Save metadata
    metadata = {
        'num_molecules': len(graphs),
        'num_smiles': len(smiles),
        'timestamp': str(torch.tensor(0).item())  # Placeholder timestamp
    }
    
    metadata_file = os.path.join(output_dir, "generation_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate molecules using trained GraphARM model')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data/ZINC',
                       help='Directory containing ZINC250k dataset (for node/edge type info)')
    
    # Generation arguments
    parser.add_argument('--num_molecules', type=int, default=10000,
                       help='Number of molecules to generate')
    parser.add_argument('--max_nodes', type=int, default=50,
                       help='Maximum number of nodes per molecule')
    parser.add_argument('--sampling_method', type=str, default='sample',
                       choices=['sample', 'argmax'],
                       help='Sampling method for generation')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./generated_molecules',
                       help='Directory to save generated molecules')
    
    # Model configuration (should match training)
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension size (must match training)')
    parser.add_argument('--num_layers', type=int, default=5,
                       help='Number of message passing layers (must match training)')
    parser.add_argument('--K', type=int, default=20,
                       help='Number of mixture components (must match training)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (must match training)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset info
    logger.info("Loading dataset information...")
    num_node_types, num_edge_types = load_dataset_info(args.data_dir)
    logger.info(f"Number of node types: {num_node_types}")
    logger.info(f"Number of edge types: {num_edge_types}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = GraphARM(
        num_node_types=num_node_types,
        num_edge_types=num_edge_types,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        K=args.K,
        dropout=args.dropout,
        device=device
    )
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    state_dict = load_model_checkpoint(args.checkpoint, device)
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Initialize masker
    dataset = ZINC(root=args.data_dir, subset='train', transform=None, pre_transform=None)
    masker = NodeMasking(dataset)
    
    # Generate molecules
    logger.info("Starting molecule generation...")
    graphs, smiles = generate_molecules(
        model=model,
        masker=masker,
        num_molecules=args.num_molecules,
        max_nodes=args.max_nodes,
        sampling_method=args.sampling_method,
        device=device
    )
    
    # Save results
    logger.info("Saving results...")
    save_results(graphs, smiles, args.output_dir)
    
    logger.info("Generation completed!")


if __name__ == '__main__':
    main()
