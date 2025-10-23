#!/usr/bin/env python3
"""
Comprehensive test script for GraphARM model.
This script tests the entire pipeline from data loading to molecule generation.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import logging
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import ZINC
import time

from models import GraphARM
from grapharm import GraphARMTrainer
from utils import NodeMasking

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_data_loading():
    """Test ZINC250k dataset loading."""
    logger.info("🧪 Testing data loading...")
    
    try:
        # Load a small subset for testing
        dataset = ZINC(root='./test_data/ZINC', subset='train', transform=None, pre_transform=None)
        logger.info(f"✅ Dataset loaded successfully: {len(dataset)} molecules")
        
        # Check data structure
        sample = dataset[0]
        logger.info(f"✅ Sample molecule: {sample.x.shape[0]} nodes, {sample.edge_index.shape[1]} edges")
        logger.info(f"✅ Node types: {sample.x.unique().shape[0]}, Edge types: {sample.edge_attr.unique().shape[0]}")
        
        return dataset, sample
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        return None, None


def test_node_masking(dataset, sample):
    """Test NodeMasking functionality."""
    logger.info("🧪 Testing NodeMasking...")
    
    try:
        masker = NodeMasking(dataset)
        
        # Test idxify
        indexed_sample = masker.idxify(sample)
        logger.info(f"✅ Idxify successful: {indexed_sample.x.shape}")
        
        # Test fully connect
        connected_sample = masker.fully_connect(indexed_sample)
        logger.info(f"✅ Fully connect successful: {connected_sample.edge_index.shape}")
        
        # Test masking
        masked_sample = masker.mask_node(connected_sample, 0)
        logger.info(f"✅ Node masking successful: {masked_sample.x[0].item()} == {masker.NODE_MASK}")
        
        # Test demasking - CORRECTED: connections_types should match number of unmasked nodes
        # After masking node 0, we have (n_nodes - 1) unmasked nodes
        unmasked_nodes = [i for i in range(masked_sample.x.shape[0]) 
                         if not masker.is_masked(masked_sample, i)]
        n_unmasked = len(unmasked_nodes)
        # Generate random edge types for connections to unmasked nodes
        edge_types = torch.randint(0, 3, (n_unmasked,))
        demasked_sample = masker.demask_node(masked_sample, 0, 1, edge_types)
        logger.info(f"✅ Node demasking successful: {demasked_sample.x[0].item()}")
        
        return masker
    except Exception as e:
        logger.error(f"❌ NodeMasking test failed: {e}")
        return None


def test_model_initialization(dataset):
    """Test model initialization."""
    logger.info("🧪 Testing model initialization...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        num_node_types = dataset.x.unique().shape[0]
        num_edge_types = dataset.edge_attr.unique().shape[0]
        
        # Initialize model
        model = GraphARM(
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            hidden_dim=64,  # Smaller for testing
            num_layers=3,   # Smaller for testing (for denoising network)
            K=5,            # Smaller for testing
            device=device
        )
        model = model.to(device)
        logger.info(f"✅ Model initialized successfully")
        logger.info(f"✅ Parameters: {sum(p.numel() for p in model.parameters())}")
        
        return model, device
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        return None, None


def test_forward_pass(model, sample, device, masker):
    """Test forward pass through the model."""
    logger.info("🧪 Testing forward pass...")
    
    try:
        model.eval()
        
        # CRITICAL: Convert sample to indexed format first using masker.idxify()
        # This maps node/edge types to 0-indexed values that match embedding sizes
        sample = masker.idxify(sample.clone())
        
        # Prepare sample - move to device and ensure long type
        sample = sample.clone()
        sample.x = sample.x.long().to(device)
        sample.edge_index = sample.edge_index.to(device)
        sample.edge_attr = sample.edge_attr.long().to(device)
        sample = sample.to(device)
        assert torch.max(sample.edge_index) < sample.x.size(0), \
            f"⚠️ Invalid edge index! max={torch.max(sample.edge_index)}, num_nodes={sample.x.size(0)}"

        with torch.no_grad():
            # Test diffusion ordering network
            ordering_probs = model.diffusion_ordering_network(sample)
            logger.info(f"✅ Ordering network output: {ordering_probs.shape}")
            
            # Test denoising network
            node_probs, edge_probs = model.denoising_network(sample)
            logger.info(f"✅ Denoising network output: nodes {node_probs.shape}, edges {edge_probs.shape}")
            
            # Test full model
            ordering_probs, node_probs, edge_probs = model(sample)
            logger.info(f"✅ Full model forward pass successful")
        
        return True
    except Exception as e:
        logger.error(f"❌ Forward pass failed: {e}")
        return False


def test_training_step(model, dataset, device):
    """Test training step."""
    logger.info("🧪 Testing training step...")
    
    try:
        # Initialize trainer
        trainer = GraphARMTrainer(
            model=model,
            dataset=dataset,
            device=device,
            learning_rate=1e-3,
            batch_size=2,  # Small batch for testing
            M=2            # Small M for testing
        )
        
        # Create small batch
        batch = [dataset[i] for i in range(min(2, len(dataset)))]
        
        # Test training step
        train_loss, val_loss = trainer.train_step(batch, batch)
        logger.info(f"✅ Training step successful: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        return trainer
    except Exception as e:
        logger.error(f"❌ Training step failed: {e}")
        return None


def test_diffusion_trajectory(trainer, sample):
    """Test diffusion trajectory generation."""
    logger.info("🧪 Testing diffusion trajectory generation...")
    
    try:
        trajectory, node_order, ordering_probs = trainer.generate_diffusion_trajectory(sample)
        logger.info(f"✅ Diffusion trajectory generated: {len(trajectory)} steps")
        logger.info(f"✅ Node order: {node_order}")
        
        # Test loss computation
        loss = trainer.compute_denoising_loss(trajectory, node_order, ordering_probs)
        logger.info(f"✅ Loss computation successful: {loss.item():.4f}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Diffusion trajectory test failed: {e}")
        return False


def test_molecule_generation(trainer):
    """Test molecule generation."""
    logger.info("🧪 Testing molecule generation...")
    
    try:
        # Generate a small molecule using trainer
        generated_graph = trainer.generate_molecule(
            max_nodes=5,  # Small molecule for testing
            sampling_method="sample"
        )
        
        logger.info(f"✅ Molecule generated: {generated_graph.x.shape[0]} nodes")
        logger.info(f"✅ Node types: {generated_graph.x.unique()}")
        logger.info(f"✅ Edge types: {generated_graph.edge_attr.unique()}")
        
        return generated_graph
    except Exception as e:
        logger.error(f"❌ Molecule generation failed: {e}")
        return None


def test_batch_generation(trainer, num_molecules=5):
    """Test batch molecule generation."""
    logger.info(f"🧪 Testing batch generation of {num_molecules} molecules...")
    
    try:
        generated_graphs = []
        generated_smiles = []
        
        start_time = time.time()
        
        for i in range(num_molecules):
            graph = trainer.generate_molecule(
                max_nodes=8,  # Small molecules for testing
                sampling_method="sample"
            )
            
            if graph is not None:
                generated_graphs.append(graph)
                # Simple SMILES placeholder
                smiles = f"C{graph.x.shape[0]}"  # Placeholder
                generated_smiles.append(smiles)
        
        end_time = time.time()
        
        logger.info(f"✅ Batch generation successful: {len(generated_graphs)} molecules")
        logger.info(f"✅ Generation time: {end_time - start_time:.2f} seconds")
        logger.info(f"✅ Average time per molecule: {(end_time - start_time) / num_molecules:.2f} seconds")
        
        return generated_graphs, generated_smiles
    except Exception as e:
        logger.error(f"❌ Batch generation failed: {e}")
        return [], []


def test_model_saving_loading(model, device):
    """Test model saving and loading."""
    logger.info("🧪 Testing model saving and loading...")
    
    try:
        # Save model
        save_path = "./test_model.pt"
        torch.save(model.state_dict(), save_path)
        logger.info(f"✅ Model saved to {save_path}")
        
        # Load model
        new_model = GraphARM(
            num_node_types=10,  # Dummy values
            num_edge_types=5,
            hidden_dim=64,
            num_layers=3,
            K=5,
            device=device
        )
        
        new_model.load_state_dict(torch.load(save_path, map_location=device))
        logger.info(f"✅ Model loaded successfully")
        
        # Clean up
        os.remove(save_path)
        logger.info(f"✅ Test file cleaned up")
        
        return True
    except Exception as e:
        logger.error(f"❌ Model saving/loading failed: {e}")
        return False


def run_comprehensive_test():
    """Run the complete test suite."""
    logger.info("🚀 Starting comprehensive GraphARM test...")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test 1: Data Loading
    dataset, sample = test_data_loading()
    test_results['data_loading'] = dataset is not None
    
    if not test_results['data_loading']:
        logger.error("❌ Data loading failed, stopping tests")
        return test_results
    
    # Test 2: NodeMasking
    masker = test_node_masking(dataset, sample)
    test_results['node_masking'] = masker is not None
    
    # Test 3: Model Initialization
    model, device = test_model_initialization(dataset)
    test_results['model_init'] = model is not None
    
    if not test_results['model_init']:
        logger.error("❌ Model initialization failed, stopping tests")
        return test_results
    
    # Test 4: Forward Pass
    test_results['forward_pass'] = test_forward_pass(model, sample, device, masker)
    
    # Test 5: Training Step
    trainer = test_training_step(model, dataset, device)
    test_results['training_step'] = trainer is not None
    
    # Test 6: Diffusion Trajectory
    if trainer:
        test_results['diffusion_trajectory'] = test_diffusion_trajectory(trainer, sample)
    
    # Test 7: Molecule Generation
    if trainer:
        test_results['molecule_generation'] = test_molecule_generation(trainer) is not None
    else:
        test_results['molecule_generation'] = False
    
    # Test 8: Batch Generation
    if trainer:
        graphs, smiles = test_batch_generation(trainer, num_molecules=3)
        test_results['batch_generation'] = len(graphs) > 0
    else:
        test_results['batch_generation'] = False
    
    # Test 9: Model Saving/Loading
    test_results['model_save_load'] = test_model_saving_loading(model, device)
    
    # Print results
    logger.info("=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY:")
    logger.info("=" * 60)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:20} : {status}")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    logger.info("=" * 60)
    logger.info(f"📈 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! GraphARM is working correctly.")
    else:
        logger.warning(f"⚠️  {total_tests - passed_tests} tests failed. Check the logs above.")
    
    return test_results


if __name__ == "__main__":
    # Run the comprehensive test
    results = run_comprehensive_test()
    
    # Exit with appropriate code
    if all(results.values()):
        exit(0)  # Success
    else:
        exit(1)  # Failure
