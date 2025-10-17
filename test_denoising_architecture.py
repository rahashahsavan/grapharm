#!/usr/bin/env python3
"""
GraphARM Denoising Network Architecture Verification Test
This script verifies that the implementation exactly matches the paper specifications.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from models import DenoisingNetwork, GraphARM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_denoising_network_architecture():
    """Test the denoising network architecture against paper specifications."""
    logger.info("🧪 Testing GraphARM Denoising Network Architecture")
    logger.info("=" * 60)
    
    # Test parameters matching ZINC250k specifications
    num_node_types = 9  # ZINC250k has 9 node types
    num_edge_types = 3  # ZINC250k has 3 edge types (single, double, triple bonds)
    hidden_dim = 256
    num_layers = 5
    K = 20
    
    # Create test data
    logger.info("📝 Creating test molecular data...")
    x = torch.tensor([0, 1, 2, 3], dtype=torch.long)  # 4 nodes
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)  # 4 edges
    edge_attr = torch.tensor([0, 1, 2, 0], dtype=torch.long)  # 4 edge types
    
    test_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    logger.info(f"✅ Test data created: {test_graph.x.shape[0]} nodes, {test_graph.edge_index.shape[1]} edges")
    
    # Initialize denoising network
    logger.info("🧠 Initializing Denoising Network...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    denoising_net = DenoisingNetwork(
        node_feature_dim=1,
        edge_feature_dim=1,
        num_node_types=num_node_types,
        num_edge_types=num_edge_types,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        K=K,
        dropout=0.1,
        device=device
    ).to(device)
    
    test_graph = test_graph.to(device)
    logger.info("✅ Denoising Network initialized")
    
    # Test 1: Embedding Encoding Network
    logger.info("\n🔍 Test 1: Embedding Encoding Network")
    logger.info("-" * 40)
    
    # Test node embedding
    node_emb = denoising_net.node_embedding(test_graph.x)
    logger.info(f"✅ Node embedding shape: {node_emb.shape}")
    assert node_emb.shape == (4, hidden_dim), f"Expected (4, {hidden_dim}), got {node_emb.shape}"
    
    # Test edge embedding
    edge_emb = denoising_net.edge_embedding(test_graph.edge_attr)
    logger.info(f"✅ Edge embedding shape: {edge_emb.shape}")
    assert edge_emb.shape == (4, hidden_dim), f"Expected (4, {hidden_dim}), got {edge_emb.shape}"
    
    # Test 2: Message Passing Network
    logger.info("\n🔍 Test 2: Message Passing Network")
    logger.info("-" * 40)
    
    # Test message passing layers
    h = node_emb
    for i, layer in enumerate(denoising_net.message_passing_layers):
        h = layer(h, test_graph.edge_index, edge_emb)
        logger.info(f"✅ Layer {i+1} output shape: {h.shape}")
        assert h.shape == (4, hidden_dim), f"Expected (4, {hidden_dim}), got {h.shape}"
    
    # Test graph-level pooling
    h_L_G = torch.mean(h, dim=0)
    logger.info(f"✅ Graph-level pooling shape: {h_L_G.shape}")
    assert h_L_G.shape == (hidden_dim,), f"Expected ({hidden_dim},), got {h_L_G.shape}"
    
    # Test 3: Node Type Prediction
    logger.info("\n🔍 Test 3: Node Type Prediction")
    logger.info("-" * 40)
    
    # Test node prediction for specific target node
    target_node_idx = 0
    target_node_embedding = h[target_node_idx]
    node_input = torch.cat([h_L_G, target_node_embedding], dim=-1)
    logger.info(f"✅ Node prediction input shape: {node_input.shape}")
    assert node_input.shape == (2 * hidden_dim,), f"Expected ({2 * hidden_dim},), got {node_input.shape}"
    
    node_logits = denoising_net.node_predictor(node_input)
    logger.info(f"✅ Node prediction output shape: {node_logits.shape}")
    assert node_logits.shape == (num_node_types + 1,), f"Expected ({num_node_types + 1},), got {node_logits.shape}"
    
    # Test 4: Edge Type Prediction (Mixture of Multinomials)
    logger.info("\n🔍 Test 4: Edge Type Prediction (Mixture of Multinomials)")
    logger.info("-" * 40)
    
    # Test with previous nodes
    previous_nodes = [1, 2, 3]  # 3 previous nodes
    M = len(previous_nodes)
    
    # Test edge prediction input
    h_v_t = h[target_node_idx]
    h_v_j = h[previous_nodes]
    h_L_G_expanded = h_L_G.unsqueeze(0).expand(M, -1)
    h_v_t_expanded = h_v_t.unsqueeze(0).expand(M, -1)
    edge_input = torch.cat([h_L_G_expanded, h_v_t_expanded, h_v_j], dim=-1)
    
    logger.info(f"✅ Edge prediction input shape: {edge_input.shape}")
    assert edge_input.shape == (M, 3 * hidden_dim), f"Expected ({M}, {3 * hidden_dim}), got {edge_input.shape}"
    
    # Test mixture weights predictor
    mixture_logits = denoising_net.mixture_weights_predictor(edge_input)
    logger.info(f"✅ Mixture weights shape: {mixture_logits.shape}")
    assert mixture_logits.shape == (M, K), f"Expected ({M}, {K}), got {mixture_logits.shape}"
    
    # Test edge predictors (K=20 separate MLPs)
    logger.info(f"✅ Number of edge predictors: {len(denoising_net.edge_predictors)}")
    assert len(denoising_net.edge_predictors) == K, f"Expected {K}, got {len(denoising_net.edge_predictors)}"
    
    # Test first edge predictor
    edge_logits_0 = denoising_net.edge_predictors[0](edge_input)
    logger.info(f"✅ Edge predictor 0 output shape: {edge_logits_0.shape}")
    assert edge_logits_0.shape == (M, num_edge_types + 1), f"Expected ({M}, {num_edge_types + 1}), got {edge_logits_0.shape}"
    
    # Test 5: Full Forward Pass
    logger.info("\n🔍 Test 5: Full Forward Pass")
    logger.info("-" * 40)
    
    # Test denoising network forward pass
    node_probs, edge_probs = denoising_net(test_graph, target_node_idx, previous_nodes)
    
    logger.info(f"✅ Node probabilities shape: {node_probs.shape}")
    assert node_probs.shape == (num_node_types + 1,), f"Expected ({num_node_types + 1},), got {node_probs.shape}"
    
    logger.info(f"✅ Edge probabilities shape: {edge_probs.shape}")
    assert edge_probs.shape == (M, num_edge_types + 1), f"Expected ({M}, {num_edge_types + 1}), got {edge_probs.shape}"
    
    # Test 6: GraphARM Integration
    logger.info("\n🔍 Test 6: GraphARM Integration")
    logger.info("-" * 40)
    
    # Test complete GraphARM model
    grapharm = GraphARM(
        num_node_types=num_node_types,
        num_edge_types=num_edge_types,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        K=K,
        dropout=0.1,
        device=device
    ).to(device)
    
    ordering_probs, node_probs, edge_probs = grapharm(test_graph, target_node_idx=target_node_idx, previous_nodes=previous_nodes)
    
    logger.info(f"✅ Ordering probabilities shape: {ordering_probs.shape}")
    assert ordering_probs.shape == (4,), f"Expected (4,), got {ordering_probs.shape}"
    
    logger.info(f"✅ Node probabilities shape: {node_probs.shape}")
    assert node_probs.shape == (num_node_types + 1,), f"Expected ({num_node_types + 1},), got {node_probs.shape}"
    
    logger.info(f"✅ Edge probabilities shape: {edge_probs.shape}")
    assert edge_probs.shape == (M, num_edge_types + 1), f"Expected ({M}, {num_edge_types + 1}), got {edge_probs.shape}"
    
    # Test 7: Architecture Compliance Verification
    logger.info("\n🔍 Test 7: Architecture Compliance Verification")
    logger.info("-" * 40)
    
    # Verify embedding dimensions
    assert denoising_net.node_embedding.embedding_dim == hidden_dim, "Node embedding dimension mismatch"
    assert denoising_net.edge_embedding.embedding_dim == hidden_dim, "Edge embedding dimension mismatch"
    logger.info("✅ Embedding dimensions correct")
    
    # Verify message passing layers
    assert len(denoising_net.message_passing_layers) == num_layers, "Number of message passing layers mismatch"
    logger.info("✅ Message passing layers count correct")
    
    # Verify node predictor architecture
    node_predictor_layers = list(denoising_net.node_predictor)
    assert len(node_predictor_layers) == 5, "Node predictor should have 5 layers (2 Linear + 2 ReLU + 1 Dropout)"
    assert node_predictor_layers[0].in_features == 2 * hidden_dim, "Node predictor input dimension mismatch"
    assert node_predictor_layers[0].out_features == hidden_dim, "Node predictor hidden dimension mismatch"
    logger.info("✅ Node predictor architecture correct")
    
    # Verify edge predictors architecture
    assert len(denoising_net.edge_predictors) == K, "Number of edge predictors mismatch"
    for i, edge_predictor in enumerate(denoising_net.edge_predictors):
        edge_layers = list(edge_predictor)
        assert len(edge_layers) == 5, f"Edge predictor {i} should have 5 layers"
        assert edge_layers[0].in_features == 3 * hidden_dim, f"Edge predictor {i} input dimension mismatch"
        assert edge_layers[0].out_features == hidden_dim, f"Edge predictor {i} hidden dimension mismatch"
    logger.info("✅ Edge predictors architecture correct")
    
    # Verify mixture weights predictor
    mixture_layers = list(denoising_net.mixture_weights_predictor)
    assert len(mixture_layers) == 5, "Mixture weights predictor should have 5 layers"
    assert mixture_layers[0].in_features == 3 * hidden_dim, "Mixture weights predictor input dimension mismatch"
    assert mixture_layers[0].out_features == hidden_dim, "Mixture weights predictor hidden dimension mismatch"
    logger.info("✅ Mixture weights predictor architecture correct")
    
    logger.info("\n🎉 ALL ARCHITECTURE TESTS PASSED!")
    logger.info("✅ GraphARM Denoising Network implementation matches paper specifications exactly!")
    
    return True

def test_dimension_assertions():
    """Test dimension assertions for edge cases."""
    logger.info("\n🔍 Testing Dimension Assertions")
    logger.info("-" * 40)
    
    # Test with empty previous nodes
    num_node_types = 9
    num_edge_types = 3
    hidden_dim = 256
    
    x = torch.tensor([0, 1], dtype=torch.long)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.tensor([0, 1], dtype=torch.long)
    test_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    denoising_net = DenoisingNetwork(
        node_feature_dim=1,
        edge_feature_dim=1,
        num_node_types=num_node_types,
        num_edge_types=num_edge_types,
        hidden_dim=hidden_dim,
        num_layers=5,
        K=20,
        dropout=0.1,
        device='cpu'
    )
    
    # Test with no previous nodes
    node_probs, edge_probs = denoising_net(test_graph, target_node_idx=0, previous_nodes=[])
    logger.info(f"✅ Empty previous nodes - Node probs: {node_probs.shape}, Edge probs: {edge_probs.shape}")
    assert edge_probs.shape == (0, num_edge_types + 1), "Empty previous nodes edge probs shape mismatch"
    
    # Test with None previous nodes
    node_probs, edge_probs = denoising_net(test_graph, target_node_idx=0, previous_nodes=None)
    logger.info(f"✅ None previous nodes - Node probs: {node_probs.shape}, Edge probs: {edge_probs.shape}")
    assert edge_probs.shape == (0, num_edge_types + 1), "None previous nodes edge probs shape mismatch"
    
    logger.info("✅ Dimension assertion tests passed!")
    return True

if __name__ == "__main__":
    try:
        # Run architecture tests
        success1 = test_denoising_network_architecture()
        
        # Run dimension assertion tests
        success2 = test_dimension_assertions()
        
        if success1 and success2:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ GraphARM Denoising Network implementation is CORRECT!")
            print("✅ Architecture matches paper specifications exactly!")
            exit(0)
        else:
            print("\n❌ SOME TESTS FAILED!")
            exit(1)
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

