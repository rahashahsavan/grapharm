#!/usr/bin/env python3
"""
End-to-end test script for GraphARM training and generation.
This script tests the complete pipeline from training to molecule generation.
"""

import torch
import torch.nn as nn
import logging
import os
import time
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ZINC

from models import GraphARM
from grapharm import GraphARMTrainer
from utils import NodeMasking

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_end_to_end():
    """Test the complete GraphARM pipeline."""
    logger.info("🚀 Starting end-to-end GraphARM test...")
    
    try:
        # 1. Load small dataset
        logger.info("📊 Loading ZINC250k dataset...")
        dataset = ZINC(root='./test_data/ZINC', subset='train', transform=None, pre_transform=None)
        
        # Use only first 10 molecules for testing
        test_dataset = dataset[:10]
        logger.info(f"✅ Dataset loaded: {len(test_dataset)} molecules")
        
        # 2. Initialize model
        logger.info("🧠 Initializing model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        num_node_types = dataset.x.unique().shape[0]
        num_edge_types = dataset.edge_attr.unique().shape[0]
        
        model = GraphARM(
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            hidden_dim=64,  # Small for testing
            num_layers=3,   # Small for testing
            K=5,            # Small for testing
            dropout=0.1,
            device=device
        )
        logger.info("✅ Model initialized")
        
        # 3. Initialize trainer
        logger.info("🏋️ Initializing trainer...")
        trainer = GraphARMTrainer(
            model=model,
            dataset=test_dataset,
            device=device,
            learning_rate=1e-3,
            batch_size=2,
            M=2
        )
        logger.info("✅ Trainer initialized")
        
        # 4. Test training
        logger.info("🏋️ Testing training...")
        train_batch = test_dataset[:2]
        val_batch = test_dataset[2:4]
        
        for epoch in range(3):  # Just 3 epochs for testing
            train_loss, val_loss = trainer.train_step(train_batch, val_batch)
            logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        logger.info("✅ Training test completed")
        
        # 5. Test molecule generation
        logger.info("🧬 Testing molecule generation...")
        
        generated_molecules = []
        start_time = time.time()
        
        for i in range(5):  # Generate 5 molecules
            molecule = trainer.generate_molecule(
                max_nodes=8,
                sampling_method="sample"
            )
            
            if molecule is not None:
                generated_molecules.append(molecule)
                logger.info(f"Generated molecule {i+1}: {molecule.x.shape[0]} nodes")
        
        end_time = time.time()
        
        logger.info(f"✅ Generated {len(generated_molecules)} molecules in {end_time - start_time:.2f} seconds")
        
        # 6. Test batch generation
        logger.info("📦 Testing batch generation...")
        
        batch_graphs, batch_smiles = [], []
        for i in range(3):
            graph = trainer.generate_molecule(max_nodes=6, sampling_method="sample")
            if graph is not None:
                batch_graphs.append(graph)
                batch_smiles.append(f"C{graph.x.shape[0]}")  # Placeholder SMILES
        
        logger.info(f"✅ Batch generation: {len(batch_graphs)} molecules")
        
        # 7. Save results
        logger.info("💾 Saving test results...")
        
        os.makedirs("./test_outputs", exist_ok=True)
        
        # Save SMILES
        with open("./test_outputs/test_smiles.txt", "w") as f:
            for smiles in batch_smiles:
                f.write(f"{smiles}\n")
        
        # Save model
        trainer.save_model("./test_outputs/test_model")
        
        logger.info("✅ Results saved to ./test_outputs/")
        
        # 8. Test model loading
        logger.info("📥 Testing model loading...")
        
        new_trainer = GraphARMTrainer(
            model=GraphARM(
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
                hidden_dim=64,
                num_layers=3,
                K=5,
                dropout=0.1,
                device=device
            ),
            dataset=test_dataset,
            device=device,
            learning_rate=1e-3,
            batch_size=2,
            M=2
        )
        
        new_trainer.load_model("./test_outputs/test_model")
        logger.info("✅ Model loading successful")
        
        # Generate with loaded model
        loaded_molecule = new_trainer.generate_molecule(max_nodes=5, sampling_method="sample")
        logger.info(f"✅ Generated with loaded model: {loaded_molecule.x.shape[0]} nodes")
        
        logger.info("🎉 END-TO-END TEST COMPLETED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        logger.error(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_end_to_end()
    if success:
        print("\n✅ GraphARM end-to-end test passed!")
        print("🚀 Ready for full training and generation!")
        exit(0)
    else:
        print("\n❌ GraphARM end-to-end test failed!")
        exit(1)
