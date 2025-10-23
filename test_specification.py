"""
Quick test to verify the specification implementation.
"""
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC
from utils import NodeMasking
from models import GraphARM
from grapharm import GraphARMTrainer

print("=" * 60)
print("🧪 Testing Specification Implementation")
print("=" * 60)

# Load dataset
print("\n1. Loading dataset...")
dataset = ZINC(root='./test_data/ZINC', subset='train', transform=None, pre_transform=None)
print(f"✅ Dataset loaded: {len(dataset)} molecules")

# Initialize
print("\n2. Initializing components...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

masker = NodeMasking(dataset)
num_node_types = dataset.x.unique().shape[0]
num_edge_types = dataset.edge_attr.unique().shape[0]

model = GraphARM(
    num_node_types=num_node_types,
    num_edge_types=num_edge_types,
    hidden_dim=64,
    num_layers=3,
    K=5,
    device=device
)

trainer = GraphARMTrainer(
    model=model,
    dataset=dataset,
    device=device,
    learning_rate=1e-3,
    batch_size=2,
    M=2
)

print("✅ Components initialized")

# Test prepare_denoising_input
print("\n3. Testing prepare_denoising_input...")
sample = dataset[0]
sample = masker.idxify(sample)
sample = masker.fully_connect(sample)
sample = sample.to(device)

n_nodes = sample.x.shape[0]
print(f"   Sample graph: {n_nodes} nodes")

# Create a mock trajectory
trajectory, node_order, _ = trainer.generate_diffusion_trajectory(sample)
print(f"   Trajectory length: {len(trajectory)}")
print(f"   Node order: {node_order[:5]}..." if len(node_order) > 5 else f"   Node order: {node_order}")

# Test at step t=2 (if possible)
if len(node_order) > 3:
    t = 2
    target_node = node_order[t]
    previous_nodes = node_order[t+1:]
    
    print(f"\n   Testing at step t={t}:")
    print(f"   - Target: {target_node}")
    print(f"   - Previous: {previous_nodes}")
    
    G_input, target_idx, previous_indices = masker.prepare_denoising_input(
        trajectory[t+1],
        target_node,
        previous_nodes
    )
    
    print(f"   ✅ Prepared graph:")
    print(f"      - Nodes: {G_input.x.shape[0]}")
    print(f"      - Target index: {target_idx}")
    print(f"      - Previous indices: {previous_indices}")
    print(f"      - Edges: {G_input.edge_index.shape[1]}")
    
    # Verify structure
    assert G_input.x.shape[0] == len(previous_nodes) + 1, "Wrong number of nodes!"
    assert target_idx == 0, "Target should be at index 0!"
    assert previous_indices == list(range(1, len(previous_nodes) + 1)), "Wrong previous indices!"
    assert G_input.x[0].item() == masker.NODE_MASK, "Target should be MASK!"
    
    print("   ✅ All assertions passed!")

# Test denoising loss computation
print("\n4. Testing denoising loss computation...")
try:
    trajectory, node_order, ordering_probs = trainer.generate_diffusion_trajectory(sample)
    loss = trainer.compute_denoising_loss(trajectory, node_order, ordering_probs, num_samples=3)
    print(f"   ✅ Loss computed: {loss.item():.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test ordering loss computation
print("\n5. Testing ordering loss computation...")
try:
    loss = trainer.compute_ordering_loss(trajectory, node_order, ordering_probs, num_samples=3)
    print(f"   ✅ Loss computed: {loss.item():.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test training step
print("\n6. Testing training step...")
try:
    batch = [dataset[i] for i in range(min(2, len(dataset)))]
    train_loss, val_loss = trainer.train_step(batch, batch)
    print(f"   ✅ Training step successful!")
    print(f"      - Train loss: {train_loss:.4f}")
    print(f"      - Val loss: {val_loss:.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ Specification Test Complete!")
print("=" * 60)


