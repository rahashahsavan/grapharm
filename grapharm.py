import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import logging
import numpy as np
from torch_geometric.data import Data, Batch

from models import GraphARM
from utils import NodeMasking

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GraphARMTrainer:
    """
    Trainer class for GraphARM model.
    Implements the training procedure as described in the paper.
    """
    def __init__(self,
                 model,
                 dataset,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 learning_rate=1e-4,
                 batch_size=32,
                 M=4):  # Number of diffusion trajectories per graph
        self.model = model.to(device)
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.M = M
        self.masker = NodeMasking(dataset)
        
        # Optimizers
        self.ordering_optimizer = torch.optim.Adam(
            self.model.diffusion_ordering_network.parameters(),
            lr=5e-2,  # CORRECTED: 5 × 10^{-2} = 0.05 per paper specification
            betas=(0.9, 0.999)
        )
        self.denoising_optimizer = torch.optim.Adam(
            self.model.denoising_network.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999)
        )
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
    
    def generate_diffusion_trajectory(self, graph):
        """
        Generate a single diffusion trajectory for a graph - CORRECTED per paper.
        
        Forward diffusion: At each step t, mask node σ_t and its edges.
        - trajectory[0] = original graph (all unmasked)
        - trajectory[t] = graph with nodes σ_0...σ_{t-1} masked
        - trajectory[n] = fully masked graph
        
        Node indices remain FIXED throughout the trajectory.
        
        Args:
            graph: PyTorch Geometric Data object
            
        Returns:
            trajectory: List of graphs [G_0, G_1, ..., G_n] where G_t has t nodes masked
            node_order: Order of nodes being masked σ = [σ_0, σ_1, ..., σ_{n-1}]
            ordering_probs: Probabilities from ordering network at each step
        """
        graph = graph.clone().to(self.device)
        graph = self.masker.idxify(graph)
        graph = self.masker.fully_connect(graph)
        
        n_nodes = graph.x.shape[0]
        
        # ASSERTION: Check valid input
        assert n_nodes > 0, "Graph must have at least one node"
        assert graph.edge_index.shape[1] == n_nodes * n_nodes, \
            f"Graph must be fully connected: expected {n_nodes * n_nodes} edges, got {graph.edge_index.shape[1]}"
        
        # Initialize trajectory with original graph
        trajectory = [graph.clone()]
        node_order = []
        ordering_probs = []
        
        current_graph = graph.clone()
        
        # Forward diffusion: mask nodes one by one
        for step in range(n_nodes):
            # Get ordering probabilities from ordering network
            # Input: current graph state + already masked nodes
            with torch.no_grad():
                probs = self.model.diffusion_ordering_network(current_graph, node_order)
                ordering_probs.append(probs)
            
            # Find unmasked nodes
            unmasked_nodes = []
            for i in range(n_nodes):
                if i not in node_order:
                    unmasked_nodes.append(i)
            
            if len(unmasked_nodes) == 0:
                break
            
            unmasked_nodes = torch.tensor(unmasked_nodes, device=self.device)
            
            # Sample next node to mask from categorical distribution
            probs_unmasked = probs[unmasked_nodes]
            probs_unmasked = probs_unmasked / (probs_unmasked.sum() + 1e-8)
            
            if len(unmasked_nodes) == 1:
                next_node = unmasked_nodes[0].item()
            else:
                dist = torch.distributions.Categorical(probs=probs_unmasked)
                sampled_idx = dist.sample()
                next_node = unmasked_nodes[sampled_idx].item()
            
            # Add to node order
            node_order.append(next_node)
            
            # Mask the selected node (and its edges to unmasked nodes)
            current_graph = self.masker.mask_node(current_graph, next_node)
            
            # Add masked graph to trajectory
            trajectory.append(current_graph.clone())
        
        # ASSERTIONS: Verify trajectory correctness
        assert len(trajectory) == n_nodes + 1, \
            f"Trajectory length mismatch: expected {n_nodes + 1}, got {len(trajectory)}"
        assert len(node_order) == n_nodes, \
            f"Node order length mismatch: expected {n_nodes}, got {len(node_order)}"
        assert len(set(node_order)) == n_nodes, \
            f"Duplicate nodes in ordering: {node_order}"
        assert all(0 <= n < n_nodes for n in node_order), \
            f"Invalid node indices in ordering: {node_order}"
        
        return trajectory, node_order, ordering_probs
    
    def compute_denoising_loss(self, trajectory, node_order, ordering_probs, num_samples=None):
        """
        Compute the denoising loss for a diffusion trajectory - CORRECTED per paper.
        
        Formula: ∑_{t} ∑_{k ∈ σ(≤t)} (n_i * w^{i,m}_k / T) * log p_θ(O_{σ(>t)} v_k | G^{i,m}_{t+1})
        
        Args:
            trajectory: List of graphs in diffusion process (forward: 0=original, n=fully masked)
            node_order: Order of nodes being absorbed (masked) in forward process
            ordering_probs: Probabilities from ordering network for each step
            num_samples: Number of timesteps T to sample (if None, use all)
            
        Returns:
            loss: Weighted denoising loss
        """
        if len(trajectory) <= 1:
            return torch.tensor(0.0, device=self.device)
        
        original_graph = trajectory[0]
        n_i = original_graph.x.shape[0]  # Number of nodes in original graph
        
        # Determine T (number of timesteps to sample)
        # CRITICAL INDEXING CLARIFICATION:
        # Paper notation (1-indexed): t ∈ {1, 2, ..., n}
        # Code notation (0-indexed): t_idx ∈ {0, 1, ..., n-1}
        # 
        # Mapping:
        # - paper t=1 → code t_idx=0: G_1 has 1 node masked (σ_0)
        # - paper t=n → code t_idx=n-1: G_n has n nodes masked (σ_0...σ_{n-1})
        #
        # Usage: trajectory[t_idx + 1] gives us G_{t_idx+1} in paper notation
        # So t_idx=0 gives trajectory[1]=G_1, t_idx=n-1 gives trajectory[n]=G_n
        #
        # Therefore: sample t_idx from [0, n-1] in code (equivalent to t ∈ [1, n] in paper)
        max_t = len(trajectory) - 1  # n (number of nodes)
        if num_samples is None:
            num_samples = max_t
        T = min(num_samples, max_t)
        
        if T == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Sample T timesteps uniformly from [0, n-1] in code indexing
        # This corresponds to paper notation t ∈ [1, n]
        if T == max_t:
            sampled_timesteps = list(range(0, max_t))  # [0, 1, ..., n-1]
        else:
            sampled_timesteps = torch.randint(0, max_t, (T,), device=self.device).tolist()
        
        total_loss = 0.0
        
        # For each sampled timestep
        for t in sampled_timesteps:
            # ASSERTION: Validate timestep range
            assert 0 <= t < max_t, f"Invalid timestep {t}: must be in [0, {max_t-1}]"
            assert t + 1 < len(trajectory), \
                f"Timestep {t} out of bounds: trajectory length {len(trajectory)}"
            
            # G_{t+1} is the graph at timestep t+1 (has t+1 nodes masked: σ_0...σ_t)
            # In reverse (denoising), we want to predict nodes that should be unmasked
            current_graph = trajectory[t + 1]
            
            # For all k ∈ σ(≤t) (all nodes masked up to and including timestep t)
            # These are nodes σ_0, σ_1, ..., σ_t (indices 0 to t in node_order)
            for k_idx in range(t + 1):
                if k_idx >= len(node_order):
                    continue
                
                k = node_order[k_idx]  # The actual node index being unmasked
                
                # Get w^{i,m}_k from ordering probabilities
                # w_k = q_φ(σ^{i,m}_t = k | G^i_0, σ^{i,m}_{<t})
                if k_idx < len(ordering_probs):
                    w_k = ordering_probs[k_idx][k].item() if k < ordering_probs[k_idx].shape[0] else 1e-8
                else:
                    w_k = 1e-8
                
                # Get predictions from denoising network
                # Network predicts node type and edges for node being unmasked
                # previous_nodes are nodes that are unmasked in current_graph (after timestep t)
                previous_nodes_list = node_order[t + 1:] if t + 1 < len(node_order) else []
                node_logits, edge_logits = self.model.denoising_network(
                    current_graph, k, previous_nodes_list
                )
                
                # Compute log probability for node type
                if k < original_graph.x.shape[0]:
                    target_node_type = original_graph.x[k].long().item()
                    node_log_prob = F.log_softmax(node_logits, dim=-1)[target_node_type]
                else:
                    node_log_prob = torch.tensor(0.0, device=self.device)
                
                # Compute log probability for edges (to previously unmasked nodes)
                edge_log_prob = torch.tensor(0.0, device=self.device)
                if edge_logits is not None and k < original_graph.x.shape[0]:
                    # Get edges from k to nodes that are unmasked (after timestep t)
                    # These are nodes at positions (t+1)..n in node_order
                    previous_nodes = node_order[t + 1:]
                    
                    for prev_idx, prev_node in enumerate(previous_nodes):
                        if prev_node >= original_graph.x.shape[0] or k >= original_graph.x.shape[0]:
                            continue
                        
                        # Find edge type between k and prev_node in original graph
                        edge_mask = ((original_graph.edge_index[0] == k) & 
                                    (original_graph.edge_index[1] == prev_node))
                        
                        if edge_mask.sum() > 0:
                            target_edge_type = original_graph.edge_attr[edge_mask][0].long().item()
                            if prev_idx < edge_logits.shape[0]:
                                edge_log_prob += F.log_softmax(edge_logits[prev_idx], dim=-1)[target_edge_type]
                
                # Total log probability for this k
                log_prob = node_log_prob + edge_log_prob
                
                # Weight by (n_i * w_k / T)
                weighted_log_prob = (n_i * w_k / T) * log_prob
                
                total_loss += weighted_log_prob
        
        # Return negative loss (we want to maximize log likelihood = minimize negative log likelihood)
        return -total_loss
    
    def compute_ordering_loss(self, trajectory, node_order, ordering_probs, num_samples=None):
        """
        Compute the ordering loss using REINFORCE algorithm - CORRECTED per paper.
        
        Reward formula: R^{i,m} = -∑_t ∑_{k ∈ σ(≤t)} (n_i/T * w^{i,m}_k) * log p_θ(...)
        REINFORCE: ∇_φ = (1/M) * ∑_{i,m} R^{i,m} * ∇ log q_φ(σ^{i,m} | G^i_0)
        
        Args:
            trajectory: List of graphs in diffusion process
            node_order: Order of nodes being absorbed
            ordering_probs: Probabilities used for ordering
            num_samples: Number of timesteps T to sample (if None, use all)
            
        Returns:
            loss: REINFORCE loss (negative because we minimize)
        """
        # Compute reward R^{i,m}
        original_graph = trajectory[0]
        n_i = original_graph.x.shape[0]
        
        # Determine T
        # INDEXING: Paper notation t ∈ [1, n] = Code notation t_idx ∈ [0, n-1]
        # See detailed explanation in compute_denoising_loss
        max_t = len(trajectory) - 1
        if num_samples is None:
            num_samples = max_t
        T = min(num_samples, max_t)
        
        if T == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Sample T timesteps uniformly from [0, n-1] (code indexing)
        if T == max_t:
            sampled_timesteps = list(range(0, max_t))  # [0, 1, ..., n-1]
        else:
            sampled_timesteps = torch.randint(0, max_t, (T,), device=self.device).tolist()
        
        # Compute reward: R^{i,m} = -∑_t ∑_k (n_i/T * w_k) * log p_θ(...)
        reward = 0.0
        
        for t in sampled_timesteps:
            # Use G_{t+1} as input
            if t + 1 >= len(trajectory):
                continue
            current_graph = trajectory[t + 1]
            
            # For all k ∈ σ(≤t)
            for k_idx in range(t + 1):
                if k_idx >= len(node_order):
                    continue
                
                k = node_order[k_idx]
                
                # Get w^{i,m}_k
                if k_idx < len(ordering_probs):
                    w_k = ordering_probs[k_idx][k].item() if k < ordering_probs[k_idx].shape[0] else 1e-8
                else:
                    w_k = 1e-8
                
                # Get predictions from denoising network (detached for reward)
                with torch.no_grad():
                    previous_nodes_list = node_order[t + 1:] if t + 1 < len(node_order) else []
                    node_logits, edge_logits = self.model.denoising_network(
                        current_graph, k, previous_nodes_list
                    )
                    
                    # Compute log probability
                    if k < original_graph.x.shape[0]:
                        target_node_type = original_graph.x[k].long().item()
                        node_log_prob = F.log_softmax(node_logits, dim=-1)[target_node_type]
                    else:
                        node_log_prob = torch.tensor(0.0, device=self.device)
                    
                    # Edge log probability
                    edge_log_prob = torch.tensor(0.0, device=self.device)
                    if edge_logits is not None and k < original_graph.x.shape[0]:
                        previous_nodes = node_order[t + 1:]
                        
                        for prev_idx, prev_node in enumerate(previous_nodes):
                            if prev_node >= original_graph.x.shape[0] or k >= original_graph.x.shape[0]:
                                continue
                            
                            edge_mask = ((original_graph.edge_index[0] == k) & 
                                        (original_graph.edge_index[1] == prev_node))
                            
                            if edge_mask.sum() > 0:
                                target_edge_type = original_graph.edge_attr[edge_mask][0].long().item()
                                if prev_idx < edge_logits.shape[0]:
                                    edge_log_prob += F.log_softmax(edge_logits[prev_idx], dim=-1)[target_edge_type]
                    
                    log_prob = node_log_prob + edge_log_prob
                
                # Accumulate reward
                reward += (n_i / T) * w_k * log_prob.item()
        
        # Reward is negative of the sum (we want high log prob = low negative log prob)
        reward = -reward
        
        # Compute log probability of the trajectory: log q_φ(σ^{i,m} | G^i_0)
        log_q_trajectory = 0.0
        for t, (probs, node_idx) in enumerate(zip(ordering_probs, node_order)):
            if node_idx < probs.shape[0]:
                log_q_trajectory += torch.log(probs[node_idx] + 1e-8)
        
        # REINFORCE loss: -R^{i,m} * log q_φ(σ^{i,m} | G^i_0)
        # Negative because we minimize the loss (gradient ascent on reward)
        ordering_loss = -reward * log_q_trajectory
        
        return ordering_loss
    
    def train_step(self, train_batch, val_batch=None):
        """
        Perform one training step - CORRECTED per paper.
        
        Key correction: Denoising network trains on train_batch,
        Ordering network trains on val_batch using REINFORCE.
        
        Args:
            train_batch: Batch of training graphs (for denoising network)
            val_batch: Batch of validation graphs (for ordering network)
            
        Returns:
            train_loss: Training loss (denoising)
            val_loss: Validation loss (ordering)
        """
        self.model.train()
        
        # ===== STEP 1: Train Denoising Network on Training Batch =====
        total_denoising_loss = 0.0
        num_train_trajectories = 0
        
        for graph in train_batch:
            # Generate M diffusion trajectories per graph
            for _ in range(self.M):
                trajectory, node_order, ordering_probs = self.generate_diffusion_trajectory(graph)
                
                # Compute denoising loss
                denoising_loss = self.compute_denoising_loss(trajectory, node_order, ordering_probs)
                
                total_denoising_loss += denoising_loss
                num_train_trajectories += 1
        
        # Normalize and update denoising network
        if num_train_trajectories > 0:
            avg_denoising_loss = total_denoising_loss / num_train_trajectories
            
            self.denoising_optimizer.zero_grad()
            avg_denoising_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.denoising_network.parameters(), max_norm=1.0)
            self.denoising_optimizer.step()
            
            train_loss = avg_denoising_loss.item()
        else:
            train_loss = 0.0
        
        # ===== STEP 2: Train Ordering Network on Validation Batch =====
        total_ordering_loss = 0.0
        num_val_trajectories = 0
        val_loss = None
        
        if val_batch is not None and len(val_batch) > 0:
            for graph in val_batch:
                # Generate M diffusion trajectories per graph
                for _ in range(self.M):
                    trajectory, node_order, ordering_probs = self.generate_diffusion_trajectory(graph)
                    
                    # Compute ordering loss with REINFORCE
                    ordering_loss = self.compute_ordering_loss(trajectory, node_order, ordering_probs)
                    
                    total_ordering_loss += ordering_loss
                    num_val_trajectories += 1
            
            # Normalize and update ordering network
            if num_val_trajectories > 0:
                avg_ordering_loss = total_ordering_loss / num_val_trajectories
                
                self.ordering_optimizer.zero_grad()
                avg_ordering_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.diffusion_ordering_network.parameters(), max_norm=1.0)
                self.ordering_optimizer.step()
                
                val_loss = avg_ordering_loss.item()
            else:
                val_loss = 0.0
        
        # Update loss tracking
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        
        # Logging
        log_dict = {
            "train_denoising_loss": train_loss,
        }
        if val_loss is not None:
            log_dict["val_ordering_loss"] = val_loss
            log_dict["total_loss"] = train_loss + val_loss
        
        wandb.log(log_dict)
        
        return train_loss, val_loss
    
    def validate(self, val_batch):
        """
        Validate the model on a batch of graphs.
        
        Args:
            val_batch: Batch of validation graphs
            
        Returns:
            val_loss: Validation loss
        """
        self.model.eval()
        
        total_loss = 0.0
        num_graphs = 0
        
        with torch.no_grad():
            for graph in val_batch:
                trajectory, node_order, ordering_probs = self.generate_diffusion_trajectory(graph)
                loss = self.compute_denoising_loss(trajectory, node_order, ordering_probs)
                total_loss += loss
                num_graphs += 1
        
        avg_loss = total_loss / num_graphs
        self.val_losses.append(avg_loss.item())
        
        wandb.log({"val_loss": avg_loss.item()})
        
        return avg_loss.item()
    
    def save_model(self, path_prefix="grapharm"):
        """
        Save the model checkpoints.
        
        Args:
            path_prefix: Prefix for checkpoint files
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ordering_optimizer_state_dict': self.ordering_optimizer.state_dict(),
            'denoising_optimizer_state_dict': self.denoising_optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, f"{path_prefix}_checkpoint.pt")
        
        torch.save(self.model.state_dict(), f"{path_prefix}_model.pt")
    
    def load_model(self, path_prefix="grapharm"):
        """
        Load the model checkpoints.
        
        Args:
            path_prefix: Prefix for checkpoint files
        """
        try:
            checkpoint = torch.load(f"{path_prefix}_checkpoint.pt", map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.ordering_optimizer.load_state_dict(checkpoint['ordering_optimizer_state_dict'])
            self.denoising_optimizer.load_state_dict(checkpoint['denoising_optimizer_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            logger.info(f"Loaded checkpoint from {path_prefix}_checkpoint.pt")
        except FileNotFoundError:
            logger.warning(f"No checkpoint found at {path_prefix}_checkpoint.pt")
    
    def generate_molecule(self, max_nodes=50, sampling_method="sample"):
        """
        Generate a single molecule using the trained model.
        
        Args:
            max_nodes: Maximum number of nodes to generate
            sampling_method: "sample" or "argmax"
            
        Returns:
            graph: Generated graph
        """
        self.model.eval()
        
        with torch.no_grad():
            # Start with a single masked node
            current_graph = self.masker.generate_fully_masked(n_nodes=1)
            current_graph = current_graph.to(self.device)
            
            for step in range(max_nodes - 1):
                # Predict next node
                node_probs, edge_probs = self.model.denoising_network(current_graph, 0)
                
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
                current_graph = self.masker.demask_node(current_graph, 0, node_type, edge_types)
                
                # Add new masked node
                current_graph = self.masker.add_masked_node(current_graph)
            
            # Demask the final node
            node_probs, edge_probs = self.model.denoising_network(current_graph, 0)
            if sampling_method == "sample":
                node_type = torch.multinomial(node_probs, 1).item()
                edge_types = torch.multinomial(edge_probs, current_graph.x.shape[0]).squeeze()
            else:
                node_type = torch.argmax(node_probs).item()
                edge_types = torch.argmax(edge_probs, dim=-1)
            
            current_graph = self.masker.demask_node(current_graph, 0, node_type, edge_types)
            
            # Remove empty edges
            current_graph = self.masker.remove_empty_edges(current_graph)
            
            return current_graph