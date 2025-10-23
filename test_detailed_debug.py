"""
Detailed Debug Test for GraphARM
Pinpoints exact location and cause of errors
"""

import torch
import traceback
import sys
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC
from utils import NodeMasking
from models import GraphARM

def print_error(test_name, e):
    """Print detailed error information"""
    print(f"\n{'='*80}")
    print(f"❌ ERROR in {test_name}")
    print(f"{'='*80}")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {str(e)}")
    print(f"\nFull Traceback:")
    print(traceback.format_exc())
    print(f"{'='*80}\n")

def test_1_data_loading():
    """Test 1: Data Loading"""
    print("\n🧪 TEST 1: Data Loading")
    try:
        dataset = ZINC(root='./test_data/ZINC', subset='train', transform=None, pre_transform=None)
        print(f"✅ Dataset loaded: {len(dataset)} molecules")
        
        sample = dataset[0]
        print(f"✅ Sample: {sample.x.shape[0]} nodes, {sample.edge_index.shape[1]} edges")
        print(f"   Node types: {sample.x.unique()}")
        print(f"   Edge types: {sample.edge_attr.unique()}")
        
        return dataset, sample
    except Exception as e:
        print_error("Data Loading", e)
        sys.exit(1)

def test_2_masker_init(dataset):
    """Test 2: NodeMasking Initialization"""
    print("\n🧪 TEST 2: NodeMasking Initialization")
    try:
        masker = NodeMasking(dataset)
        print(f"✅ Masker initialized")
        print(f"   NODE_MASK: {masker.NODE_MASK}")
        print(f"   EDGE_MASK: {masker.EDGE_MASK}")
        print(f"   Num node types: {masker.num_node_types}")
        print(f"   Num edge types: {masker.num_edge_types}")
        
        return masker
    except Exception as e:
        print_error("Masker Initialization", e)
        sys.exit(1)

def test_3_model_init(masker, device):
    """Test 3: Model Initialization"""
    print("\n🧪 TEST 3: Model Initialization")
    try:
        model = GraphARM(
            num_node_types=masker.num_node_types,
            num_edge_types=masker.num_edge_types,
            hidden_dim=64,
            num_layers=5,
            K=10,
            device=device
        ).to(device)
        
        print(f"✅ Model initialized on {device}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters())}")
        
        return model
    except Exception as e:
        print_error("Model Initialization", e)
        sys.exit(1)

def test_4_diffusion_ordering_network(model, sample, masker, device):
    """Test 4: Diffusion Ordering Network"""
    print("\n🧪 TEST 4: Diffusion Ordering Network")
    try:
        # Prepare sample
        print("   Preparing sample...")
        sample = masker.idxify(sample.clone())
        sample = sample.to(device)
        sample.x = sample.x.long()
        sample.edge_attr = sample.edge_attr.long()
        
        print(f"   Sample prepared: {sample.x.shape}, {sample.edge_index.shape}")
        print(f"   Node types range: [{sample.x.min()}, {sample.x.max()}]")
        print(f"   Edge types range: [{sample.edge_attr.min()}, {sample.edge_attr.max()}]")
        
        # Test ordering network
        print("   Testing ordering network...")
        model.eval()
        with torch.no_grad():
            ordering_probs = model.diffusion_ordering_network(sample)
        
        print(f"✅ Ordering network output: {ordering_probs.shape}")
        print(f"   Probs range: [{ordering_probs.min():.4f}, {ordering_probs.max():.4f}]")
        print(f"   Probs sum: {ordering_probs.sum():.4f}")
        
        return sample
    except Exception as e:
        print_error("Diffusion Ordering Network", e)
        print(f"   Sample info at error:")
        print(f"     x.shape: {sample.x.shape if hasattr(sample, 'x') else 'N/A'}")
        print(f"     edge_index.shape: {sample.edge_index.shape if hasattr(sample, 'edge_index') else 'N/A'}")
        print(f"     edge_attr.shape: {sample.edge_attr.shape if hasattr(sample, 'edge_attr') else 'N/A'}")
        sys.exit(1)

def test_5_denoising_network(model, sample, device):
    """Test 5: Denoising Network"""
    print("\n🧪 TEST 5: Denoising Network")
    try:
        print("   Testing denoising network (no target_node)...")
        model.eval()
        with torch.no_grad():
            node_probs, edge_probs = model.denoising_network(sample)
        
        print(f"✅ Denoising network output:")
        print(f"   Node probs: {node_probs.shape if node_probs is not None else 'None'}")
        print(f"   Edge probs: {edge_probs.shape if edge_probs is not None else 'None'}")
        
        # Test with target node
        print("\n   Testing denoising network (with target_node=0)...")
        with torch.no_grad():
            node_probs, edge_probs = model.denoising_network(
                sample, 
                target_node_idx=0, 
                previous_nodes=[1, 2, 3]
            )
        
        print(f"✅ Denoising network output (with target):")
        print(f"   Node probs: {node_probs.shape if node_probs is not None else 'None'}")
        print(f"   Edge probs: {edge_probs.shape if edge_probs is not None else 'None'}")
        
        return True
    except Exception as e:
        print_error("Denoising Network", e)
        print(f"   Sample info at error:")
        print(f"     x.shape: {sample.x.shape}")
        print(f"     edge_index.shape: {sample.edge_index.shape}")
        print(f"     edge_attr.shape: {sample.edge_attr.shape}")
        print(f"     device: {sample.x.device}")
        sys.exit(1)

def test_6_trajectory_generation(dataset, masker, model, device):
    """Test 6: Diffusion Trajectory Generation"""
    print("\n🧪 TEST 6: Diffusion Trajectory Generation")
    try:
        from grapharm import GraphARMTrainer
        
        print("   Initializing trainer...")
        trainer = GraphARMTrainer(
            model=model,
            dataset=dataset,
            device=device,
            learning_rate=1e-3,
            batch_size=1,
            M=2
        )
        
        print("   Generating trajectory...")
        graph = dataset[0].clone()
        trajectory, node_order, ordering_probs = trainer.generate_diffusion_trajectory(graph)
        
        print(f"✅ Trajectory generated:")
        print(f"   Trajectory length: {len(trajectory)}")
        print(f"   Node order length: {len(node_order)}")
        print(f"   Node order: {node_order[:10]}..." if len(node_order) > 10 else f"   Node order: {node_order}")
        
        # Check each trajectory step
        print(f"\n   Trajectory details:")
        for i in range(min(5, len(trajectory))):
            g = trajectory[i]
            n_masked = (g.x == masker.NODE_MASK).sum().item()
            print(f"     t={i}: {g.x.shape[0]} nodes, {g.edge_index.shape[1]} edges, {n_masked} masked")
        
        return trainer, trajectory, node_order, ordering_probs
    except Exception as e:
        print_error("Trajectory Generation", e)
        sys.exit(1)

def test_7_prepare_denoising_input(trainer, trajectory, node_order, masker):
    """Test 7: Prepare Denoising Input"""
    print("\n🧪 TEST 7: Prepare Denoising Input")
    try:
        print(f"   Testing prepare_denoising_input for different steps...")
        
        # Test multiple steps
        for t in [0, len(node_order)//2, len(node_order)-2]:
            if t >= len(node_order) - 1:
                continue
                
            print(f"\n   Step t={t}:")
            G_t_plus_1 = trajectory[t + 1]
            target_node = node_order[t]
            previous_nodes = node_order[t + 1:]
            
            print(f"     G_t_plus_1: {G_t_plus_1.x.shape[0]} nodes, {G_t_plus_1.edge_index.shape[1]} edges")
            print(f"     target_node: {target_node}")
            print(f"     previous_nodes ({len(previous_nodes)}): {previous_nodes[:5]}..." if len(previous_nodes) > 5 else f"     previous_nodes: {previous_nodes}")
            
            # Validate indices
            print(f"     Validating indices...")
            print(f"       target_node < n_nodes: {target_node} < {G_t_plus_1.x.shape[0]} = {target_node < G_t_plus_1.x.shape[0]}")
            valid_prev = [n for n in previous_nodes if n < G_t_plus_1.x.shape[0]]
            print(f"       Valid previous_nodes: {len(valid_prev)}/{len(previous_nodes)}")
            if len(valid_prev) != len(previous_nodes):
                invalid = [n for n in previous_nodes if n >= G_t_plus_1.x.shape[0]]
                print(f"       ⚠️  INVALID indices: {invalid}")
            
            # Prepare input
            print(f"     Calling prepare_denoising_input...")
            try:
                G_input, target_idx, previous_indices = masker.prepare_denoising_input(
                    G_t_plus_1,
                    target_node,
                    previous_nodes
                )
                
                print(f"     ✅ Result:")
                print(f"        G_input: {G_input.x.shape[0]} nodes, {G_input.edge_index.shape[1]} edges")
                print(f"        target_idx: {target_idx}")
                print(f"        previous_indices: {previous_indices[:5]}..." if len(previous_indices) > 5 else f"        previous_indices: {previous_indices}")
                
            except Exception as sub_e:
                print(f"     ❌ ERROR in prepare_denoising_input at t={t}:")
                print(f"        {type(sub_e).__name__}: {sub_e}")
                print(f"        Traceback:")
                print(traceback.format_exc())
                raise sub_e
        
        print(f"\n✅ Prepare denoising input test passed")
        return True
    except Exception as e:
        print_error("Prepare Denoising Input", e)
        sys.exit(1)

def test_8_compute_denoising_loss(trainer, trajectory, node_order, ordering_probs):
    """Test 8: Compute Denoising Loss"""
    print("\n🧪 TEST 8: Compute Denoising Loss")
    try:
        print("   Computing denoising loss...")
        loss = trainer.compute_denoising_loss(trajectory, node_order, ordering_probs)
        
        print(f"✅ Denoising loss computed:")
        print(f"   Loss: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print_error("Compute Denoising Loss", e)
        print(f"   Trajectory info:")
        print(f"     Length: {len(trajectory)}")
        print(f"     Node order: {node_order}")
        sys.exit(1)

def test_9_full_training_step(trainer, dataset):
    """Test 9: Full Training Step"""
    print("\n🧪 TEST 9: Full Training Step")
    try:
        print("   Creating mini batch...")
        batch = [dataset[i] for i in range(min(2, len(dataset)))]
        
        print("   Running training step...")
        train_loss, val_loss = trainer.train_step(batch, batch)
        
        print(f"✅ Training step completed:")
        print(f"   Train loss: {train_loss:.4f}")
        print(f"   Val loss: {val_loss:.4f}")
        
        return True
    except Exception as e:
        print_error("Full Training Step", e)
        sys.exit(1)

def main():
    """Run all debug tests"""
    print("\n" + "="*80)
    print("DETAILED DEBUG TEST FOR GRAPHARM")
    print("="*80)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # Run tests
    dataset, sample = test_1_data_loading()
    masker = test_2_masker_init(dataset)
    model = test_3_model_init(masker, device)
    sample = test_4_diffusion_ordering_network(model, sample, masker, device)
    test_5_denoising_network(model, sample, device)
    trainer, trajectory, node_order, ordering_probs = test_6_trajectory_generation(dataset, masker, model, device)
    test_7_prepare_denoising_input(trainer, trajectory, node_order, masker)
    test_8_compute_denoising_loss(trainer, trajectory, node_order, ordering_probs)
    test_9_full_training_step(trainer, dataset)
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

