#!/usr/bin/env python3
"""
Quick test script for GraphARM model.
This script performs a minimal test to verify the model works.
"""

import torch
import logging
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC

from models import GraphARM
from utils import NodeMasking

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def quick_test():
    """Quick test of GraphARM components."""
    logger.info("🚀 Starting quick GraphARM test...")
    
    try:
        # 1. Create dummy data
        logger.info("📝 Creating dummy molecular data...")
        x = torch.tensor([[0], [1], [2]], dtype=torch.long)  # 3 nodes
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  # 3 edges
        edge_attr = torch.tensor([0, 1, 2], dtype=torch.long)  # 3 edge types
        
        dummy_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        logger.info(f"✅ Dummy data created: {dummy_data.x.shape[0]} nodes, {dummy_data.edge_index.shape[1]} edges")
        
        # 2. Initialize masker
        logger.info("🎭 Initializing NodeMasking...")
        masker = NodeMasking(dummy_data)
        logger.info("✅ NodeMasking initialized")
        
        # 3. Initialize model
        logger.info("🧠 Initializing GraphARM model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        model = GraphARM(
            num_node_types=3,
            num_edge_types=3,
            hidden_dim=32,  # Small for quick test
            num_layers=2,   # Small for quick test
            K=3,            # Small for quick test
            device=device
        )
        logger.info("✅ Model initialized")
        
        # 4. Test forward pass
        logger.info("⚡ Testing forward pass...")
        model.eval()
        
        with torch.no_grad():
            # Test ordering network
            ordering_probs = model.diffusion_ordering_network(dummy_data)
            logger.info(f"✅ Ordering network: {ordering_probs.shape}")
            
            # Test denoising network with target node and previous nodes
            target_node_idx = 0
            previous_nodes = [1, 2]  # Previous nodes for edge prediction
            node_probs, edge_probs = model.denoising_network(dummy_data, target_node_idx, previous_nodes)
            logger.info(f"✅ Denoising network: nodes {node_probs.shape}, edges {edge_probs.shape if edge_probs is not None else 'None'}")
        
        # 5. Test masking operations
        logger.info("🎭 Testing masking operations...")
        
        # Fully connect the graph
        connected_data = masker.fully_connect(dummy_data)
        logger.info(f"✅ Fully connected: {connected_data.edge_index.shape[1]} edges")
        
        # Mask a node
        masked_data = masker.mask_node(connected_data, 0)
        logger.info(f"✅ Node masked: {masked_data.x[0].item()}")
        
        # 6. Test complete forward pass
        logger.info("🔄 Testing complete forward pass...")
        ordering_probs, node_probs, edge_probs = model(dummy_data, target_node_idx=0, previous_nodes=[1, 2])
        logger.info(f"✅ Complete forward pass successful")
        logger.info(f"   - Ordering probs: {ordering_probs.shape}")
        logger.info(f"   - Node probs: {node_probs.shape}")
        logger.info(f"   - Edge probs: {edge_probs.shape if edge_probs is not None else 'None'}")
        
        logger.info("🎉 ALL QUICK TESTS PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✅ GraphARM is working correctly!")
        exit(0)
    else:
        print("\n❌ GraphARM test failed!")
        exit(1)
