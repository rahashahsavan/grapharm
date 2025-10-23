import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import RGCNConv
from torch.nn import functional as F
from torch.nn import Linear, ReLU
import torch.nn.init as init
import math


class GraphARMMessagePassing(MessagePassing):
    """
    CORRECTED: Custom message passing layer for GraphARM - EXACT per paper specifications.
    
    Paper specifications:
    - Message MLP f: 2-layer, ReLU, hidden size 256, NO dropout
    - Attention MLP g: 2-layer, ReLU, hidden size 256, NO dropout
    - Input: [h_l^{v_i} || h_l^{v_j} || h_e^{v_i,v_j}]
    - Attention: sigmoid(g(...))
    - Update: GRU(h_l^{v_i}, sum_{j} a_{i,j}^l * m_{i,j}^l)
    """
    def __init__(self, hidden_dim):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        
        # Step 2.1: Message function f - 2-layer MLP with ReLU, hidden size 256
        self.message_mlp = nn.Sequential(
            Linear(3 * hidden_dim, hidden_dim),  # [h_i || h_j || h_e] -> 256
            ReLU(),
            Linear(hidden_dim, hidden_dim)       # 256 -> 256
        )
        
        # Step 2.2: Attention function g - 2-layer MLP with ReLU, hidden size 256
        self.attention_mlp = nn.Sequential(
            Linear(3 * hidden_dim, hidden_dim),  # [h_i || h_j || h_e] -> 256
            ReLU(),
            Linear(hidden_dim, 1)                # 256 -> 1
        )
        
        # Step 2.3: GRU for node embedding update
        # GRU(h_l^{v_i}, aggregated_messages)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass - EXACT implementation per paper.
        
        Args:
            x: Node features [N, hidden_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge attributes [E, hidden_dim]
        
        Returns:
            Updated node features [N, hidden_dim]
        """
        # Step 2.1 & 2.2: Compute messages and attention, then aggregate
        aggregated = self.propagate(edge_index, x=x, edge_attr=edge_attr)  # [N, hidden_dim]
        
        # Step 2.3: Update with GRU
        # h_{l+1}^{v_i} = GRU(h_l^{v_i}, sum_{j in N(i)} a_{i,j}^l * m_{i,j}^l)
        out = self.gru(aggregated, x)  # GRU(input, hidden)
        
        return out
    
    def message(self, x_i, x_j, edge_attr):
        """
        Compute messages and attention - EXACT per paper.
        
        Step 2.1: m_{i,j}^l = f([h_l^{v_i} || h_l^{v_j} || h_e^{v_i,v_j}])
        Step 2.2: a_{i,j}^l = sigmoid(g([h_l^{v_i} || h_l^{v_j} || h_e^{v_i,v_j}]))
        
        Returns: a_{i,j}^l * m_{i,j}^l
        """
        # Concatenate: [h_l^{v_i} || h_l^{v_j} || h_e^{v_i,v_j}]
        edge_features = torch.cat([x_i, x_j, edge_attr], dim=-1)  # [E, 3*hidden_dim]
        
        # Step 2.1: Compute message m_{i,j}^l
        message = self.message_mlp(edge_features)  # [E, hidden_dim]
        
        # Step 2.2: Compute attention a_{i,j}^l
        attention = self.attention_mlp(edge_features)  # [E, 1]
        attention = torch.sigmoid(attention)            # sigmoid activation
        
        # Return weighted message: a_{i,j}^l * m_{i,j}^l
        return message * attention


class DenoisingNetwork(nn.Module):
    """
    CORRECTED: Denoising Network for GraphARM - EXACT per paper specifications for ZINC250k.
    
    Architecture per paper:
    1. Embedding Encoding: Single-layer MLP for nodes and edges
    2. Message Passing: L=5 layers with 2-layer MLPs f and g (hidden 256), GRU update
    3. Graph Pooling: Average pooling to get h_L^G
    4. Node Prediction: 2-layer MLP_n with ReLU, hidden 256
    5. Edge Prediction: K=20 separate 2-layer MLPs with ReLU, hidden 256, mixture model
    
    ALL MLPs: 2-layer, ReLU activation, hidden dimension 256, NO dropout
    """
    def __init__(self,
                 num_node_types,
                 num_edge_types,
                 hidden_dim=256,
                 num_layers=5,
                 K=20,
                 device='cpu'):
        super(DenoisingNetwork, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.K = K
        
        # Add mask token and "no edge" type
        self.num_node_types = num_node_types + 1  # +1 for mask token
        self.num_edge_types = num_edge_types + 1  # +1 for "no edge"
        
        # Step 1: Embedding Encoding Network
        # h_0^{v_i} = MLP(v_i) - single-layer linear
        self.node_embedding = nn.Embedding(self.num_node_types, hidden_dim)
        
        # h_e^{v_i, v_j} = MLP(e_{v_i, v_j}) - single-layer linear
        self.edge_embedding = nn.Embedding(self.num_edge_types, hidden_dim)
        
        # Step 2: Message Passing Network (L=5 layers)
        self.message_passing_layers = nn.ModuleList([
            GraphARMMessagePassing(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Step 3: Node Type Prediction
        # MLP_n([h_L^G || h_{v_{\tilde{τ}_t}}]) - 2-layer MLP with ReLU, hidden 256
        self.node_predictor = nn.Sequential(
            Linear(2 * hidden_dim, hidden_dim),      # [h_L^G || h_v_t] -> 256
            ReLU(),
            Linear(hidden_dim, self.num_node_types)  # 256 -> num_node_types
        )
        
        # Step 4: Edge Type Prediction - Mixture of K=20 Multinomials
        # K=20 separate MLPs: MLP_{e_1}, ..., MLP_{e_20}
        # Each: 2-layer MLP with ReLU, hidden 256
        # Input: [h_L^G || h_{v_t} || h_{v_j}]
        # Output: E+1 (edge types + "no edge")
        self.edge_predictors = nn.ModuleList([
            nn.Sequential(
                Linear(3 * hidden_dim, hidden_dim),      # [h_L^G || h_v_t || h_v_j] -> 256
                ReLU(),
                Linear(hidden_dim, self.num_edge_types)  # 256 -> E+1
            )
            for _ in range(K)
        ])
        
        # Mixture weights predictor MLP_α
        # 2-layer MLP with ReLU, hidden 256
        # Input: [h_L^G || h_{v_t} || h_{v_j}]
        # Output: K weights
        self.mixture_weights_predictor = nn.Sequential(
            Linear(3 * hidden_dim, hidden_dim),  # [h_L^G || h_v_t || h_v_j] -> 256
            ReLU(),
            Linear(hidden_dim, K)                # 256 -> K=20
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        for module in self.modules():
            if isinstance(module, Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, graph, target_node_idx=None, previous_nodes=None):
        """
        Forward pass - EXACT implementation per paper specifications.
        
        Args:
            graph: PyTorch Geometric Data object
            target_node_idx: Index of node being denoised at step t
            previous_nodes: List of previously generated nodes (after step t)
        
        Returns:
            node_probs: p(v_{\tilde{τ}_t} | G_{t+1})
            edge_probs: p(e_{v_t, v_j} | G_{t+1}) for all previous nodes v_j
        """
        # Get graph data - ENSURE all tensors are on correct device
        x = graph.x.long().squeeze(-1).to(self.device)  # [N]
        edge_index = graph.edge_index.to(self.device)   # [2, E]
        edge_attr = graph.edge_attr.long().squeeze(-1).to(self.device)  # [E]
        
        # Step 1: Embedding Encoding
        h = self.node_embedding(x)         # [N, hidden_dim]
        edge_emb = self.edge_embedding(edge_attr)  # [E, hidden_dim]
        
        # Step 2: Message Passing (L=5 layers)
        for layer in self.message_passing_layers:
            h = layer(h, edge_index, edge_emb)
        # After L=5 layers: h_L^{v_i}
        
        # Step 2.4: Graph-Level Pooling
        # h_L^G = AveragePool(h_L^{v_i} for all nodes i)
        h_L_G = torch.mean(h, dim=0)  # [hidden_dim]
        
        # Step 3: Node Type Prediction
        # p(v_{\tilde{τ}_t} | G_{t+1}) = Softmax(MLP_n([h_L^G || h_{v_{\tilde{τ}_t}}]))
        if target_node_idx is not None:
            # CRITICAL FIX: Validate target_node_idx is within bounds
            if target_node_idx >= h.shape[0]:
                raise IndexError(f"target_node_idx {target_node_idx} is out of bounds for tensor with {h.shape[0]} nodes")
            # Predict for specific target node
            target_emb = h[target_node_idx]  # [hidden_dim]
            node_input = torch.cat([h_L_G, target_emb], dim=-1)  # [2*hidden_dim]
            node_logits = self.node_predictor(node_input)  # [num_node_types]
            node_probs = F.softmax(node_logits, dim=-1)
        else:
            # Predict for all nodes
            h_L_G_expanded = h_L_G.unsqueeze(0).expand(h.size(0), -1)  # [N, hidden_dim]
            node_input = torch.cat([h_L_G_expanded, h], dim=-1)  # [N, 2*hidden_dim]
            node_logits = self.node_predictor(node_input)  # [N, num_node_types]
            node_probs = F.softmax(node_logits, dim=-1)
        
        # Step 4: Edge Type Prediction (Mixture of Multinomials)
        edge_probs = None
        if previous_nodes is not None and len(previous_nodes) > 0 and target_node_idx is not None:
            # CRITICAL FIX: Validate all indices are within bounds
            if target_node_idx >= h.shape[0]:
                raise IndexError(f"target_node_idx {target_node_idx} is out of bounds for tensor with {h.shape[0]} nodes")
            
            # Filter previous_nodes to only include valid indices (safety check)
            previous_nodes = [n for n in previous_nodes if n < h.shape[0]]
            if len(previous_nodes) == 0:
                return node_probs, None
            
            M = len(previous_nodes)  # Number of previous nodes
            
            # Current node embedding
            h_v_t = h[target_node_idx]  # [hidden_dim]
            
            # Previous nodes embeddings - CRITICAL: Convert to tensor if list
            if isinstance(previous_nodes, list):
                previous_nodes_tensor = torch.tensor(previous_nodes, dtype=torch.long, device=self.device)
            else:
                previous_nodes_tensor = previous_nodes
            h_v_j = h[previous_nodes_tensor]  # [M, hidden_dim]
            
            # Expand for all previous nodes
            h_L_G_expanded = h_L_G.unsqueeze(0).expand(M, -1)  # [M, hidden_dim]
            h_v_t_expanded = h_v_t.unsqueeze(0).expand(M, -1)  # [M, hidden_dim]
            
            # Input: [h_L^G || h_{v_t} || h_{v_j}]
            edge_input = torch.cat([h_L_G_expanded, h_v_t_expanded, h_v_j], dim=-1)  # [M, 3*hidden_dim]
            
            # Compute mixture weights
            # [α_1, ..., α_K] = Softmax(MLP_α([h_L^G || h_{v_t} || h_{v_j}]))
            mixture_logits = self.mixture_weights_predictor(edge_input)  # [M, K]
            mixture_weights = F.softmax(mixture_logits, dim=-1)  # [M, K]
            
            # Compute edge logits for each component k
            # logits_{k,j} = MLP_{e_k}([h_L^G || h_{v_t} || h_{v_j}])
            edge_logits_all = []
            for k in range(self.K):
                logits_k = self.edge_predictors[k](edge_input)  # [M, num_edge_types]
                edge_logits_all.append(logits_k)
            
            edge_logits_all = torch.stack(edge_logits_all, dim=0)  # [K, M, num_edge_types]
            
            # Final probability: p(e_{v_t, v_j} | G_{t+1}) = sum_{k=1}^K α_k * Softmax(logits_{k,j})
            edge_probs_per_component = F.softmax(edge_logits_all, dim=-1)  # [K, M, num_edge_types]
            
            # Weight by mixture weights
            mixture_weights_expanded = mixture_weights.t().unsqueeze(-1)  # [K, M, 1]
            edge_probs = torch.sum(mixture_weights_expanded * edge_probs_per_component, dim=0)  # [M, num_edge_types]
        
        return node_probs, edge_probs


class DiffusionOrderingNetwork(nn.Module):
    """
    Diffusion Ordering Network for GraphARM - EXACT per paper specification.
    
    Paper specification (Page 16, Section A.8):
    "Diffusion ordering network: We use a 3-layer relational graph convolutional 
    network with hidden dimension 256."
    
    CRITICAL: Must use RGCN with 3 layers, not custom message passing with 5 layers.
    """
    def __init__(self,
                 num_node_types,
                 num_edge_types,
                 hidden_dim=256,
                 num_layers=3,  # MUST be 3 per paper specification
                 device='cpu'):
        super(DiffusionOrderingNetwork, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Add mask token and empty edge type
        self.num_node_types = num_node_types + 1  # +1 for mask token
        self.num_edge_types = num_edge_types + 2  # +1 for mask token, +1 for empty edge
        
        # Node embedding
        self.node_embedding = nn.Embedding(self.num_node_types, hidden_dim)
        
        # 3-layer RGCN (Relational Graph Convolutional Network)
        # Paper: "3-layer relational graph convolutional network"
        self.rgcn_layers = nn.ModuleList()
        
        # Layer 1: hidden_dim -> hidden_dim
        self.rgcn_layers.append(
            RGCNConv(hidden_dim, hidden_dim, self.num_edge_types)
        )
        
        # Layer 2: hidden_dim -> hidden_dim
        self.rgcn_layers.append(
            RGCNConv(hidden_dim, hidden_dim, self.num_edge_types)
        )
        
        # Layer 3: hidden_dim -> hidden_dim
        self.rgcn_layers.append(
            RGCNConv(hidden_dim, hidden_dim, self.num_edge_types)
        )
        
        # Output layer for node selection probabilities
        # 2-layer MLP with ReLU, hidden size 256
        self.output_layer = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1)
        )
        
        # Positional encoding for absorbed nodes
        # CORRECTED: Use register_buffer to ensure pos_encoding moves with model
        self.max_nodes = 100
        self.register_buffer('pos_encoding', self._create_positional_encoding(self.max_nodes, hidden_dim))
        
        self.reset_parameters()
    
    def _create_positional_encoding(self, max_len, d_model):
        """Create positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def reset_parameters(self):
        """Initialize parameters."""
        for module in self.modules():
            if isinstance(module, Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, graph, node_order=None):
        """
        Forward pass - EXACT per paper specification.
        
        Args:
            graph: PyTorch Geometric Data object
            node_order: List of already absorbed nodes
        
        Returns:
            node_probs: Probabilities for selecting each node [N]
        """
        if node_order is None:
            node_order = []
        
        # Get graph data - ENSURE all tensors are on correct device
        x = graph.x.long().squeeze(-1).to(self.device)  # [N]
        edge_index = graph.edge_index.to(self.device)   # [2, E]
        edge_type = graph.edge_attr.long().squeeze(-1).to(self.device)  # [E]
        
        # Embed nodes
        h = self.node_embedding(x)  # [N, hidden_dim]
        
        # Add positional encoding for absorbed nodes
        for i, node_idx in enumerate(node_order):
            if i < self.max_nodes and node_idx < h.size(0):
                h[node_idx] += self.pos_encoding[i]  # No need for .to(device) - buffer handles it
        
        # 3-layer RGCN message passing
        for layer in self.rgcn_layers:
            h = layer(h, edge_index, edge_type)
            h = F.relu(h)  # ReLU activation after each RGCN layer
        
        # Compute node selection probabilities
        logits = self.output_layer(h).squeeze(-1)  # [N]
        
        # Mask already absorbed nodes
        mask = torch.ones_like(logits, dtype=torch.bool)
        for node_idx in node_order:
            if node_idx < mask.size(0):
                mask[node_idx] = False
        
        # Apply mask and softmax
        masked_logits = logits.masked_fill(~mask, float('-inf'))
        node_probs = F.softmax(masked_logits, dim=0)
        
        return node_probs


class GraphARM(nn.Module):
    """
    Complete GraphARM model - CORRECTED implementation per paper specifications.
    
    CRITICAL FIXES:
    - Diffusion Ordering Network: 3-layer RGCN (not 5-layer custom message passing)
    - Denoising Network: 5-layer custom message passing with GRUCell
    """
    def __init__(self,
                 num_node_types,
                 num_edge_types,
                 hidden_dim=256,
                 num_layers=5,  # For denoising network only
                 K=20,
                 device='cpu'):
        super(GraphARM, self).__init__()
        self.device = device
        
        # Diffusion Ordering Network: 3-layer RGCN per paper (Page 16, A.8)
        self.diffusion_ordering_network = DiffusionOrderingNetwork(
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            hidden_dim=hidden_dim,
            num_layers=3,  # MUST be 3 per paper specification
            device=device
        )
        
        # Denoising Network: 5-layer custom message passing per paper
        self.denoising_network = DenoisingNetwork(
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            hidden_dim=hidden_dim,
            num_layers=num_layers,  # 5 layers for denoising
            K=K,
            device=device
        )
    
    def forward(self, graph, node_order=None, target_node_idx=None, previous_nodes=None):
        ordering_probs = self.diffusion_ordering_network(graph, node_order)
        node_probs, edge_probs = self.denoising_network(graph, target_node_idx, previous_nodes)
        return ordering_probs, node_probs, edge_probs