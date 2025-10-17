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
            lr=5 * 1e-5 ,
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
        Generate a single diffusion trajectory for a graph.
        
        Args:
            graph: PyTorch Geometric Data object
            
        Returns:
            trajectory: List of graphs in the diffusion process
            node_order: Order of nodes being absorbed
            ordering_probs: Probabilities used for ordering
        """
        graph = graph.clone().to(self.device)
        graph = self.masker.idxify(graph)
        graph = self.masker.fully_connect(graph)
        
        trajectory = [graph.clone()]
        node_order = []
        ordering_probs = []
        
        current_graph = graph.clone()
        
        for step in range(graph.x.shape[0]):
            # Get ordering probabilities
            with torch.no_grad():
                probs = self.model.diffusion_ordering_network(current_graph, node_order)
                ordering_probs.append(probs)
            
            # Sample next node to absorb
            unmasked_nodes = torch.tensor([i for i in range(current_graph.x.shape[0]) 
                                         if i not in node_order], device=self.device)
            
            if len(unmasked_nodes) == 0:
                break
                
            # Sample from categorical distribution
            probs_unmasked = probs[unmasked_nodes]
            probs_unmasked = probs_unmasked / probs_unmasked.sum()
            
            if len(unmasked_nodes) == 1:
                next_node = unmasked_nodes[0]
            else:
                dist = torch.distributions.Categorical(probs=probs_unmasked)
                sampled_idx = dist.sample()
                next_node = unmasked_nodes[sampled_idx]
            
            node_order.append(next_node.item())
            
            # Mask the selected node
            masked_graph = self.masker.mask_node(current_graph, next_node.item())
            trajectory.append(masked_graph.clone())
            
            # Remove the masked node for next iteration
            if step < graph.x.shape[0] - 1:
                current_graph = self.masker.remove_node(masked_graph, next_node.item())
                # Update node indices in node_order
                node_order = [idx - 1 if idx > next_node.item() else idx for idx in node_order]
        
        return trajectory, node_order, ordering_probs
    
    def compute_denoising_loss(self, trajectory, node_order, ordering_probs):
        """
        Compute the denoising loss for a diffusion trajectory.
        
        Args:
            trajectory: List of graphs in diffusion process
            node_order: Order of nodes being absorbed
            ordering_probs: Probabilities used for ordering
            
        Returns:
            loss: Denoising loss
        """
        if len(trajectory) <= 1:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = 0.0
        original_graph = trajectory[0]
        
        # Reverse the trajectory for denoising
        for t in range(len(trajectory) - 1, 0, -1):
            current_graph = trajectory[t]
            target_node_idx = len(trajectory) - 1 - t
            
            if target_node_idx >= len(node_order):
                continue
                
            target_node = node_order[target_node_idx]
            
            # Get predictions from denoising network
            node_probs, edge_probs = self.model.denoising_network(current_graph, target_node_idx)
            
            # Node loss
            if target_node < original_graph.x.shape[0]:
                target_node_type = original_graph.x[target_node].long().item()
                node_loss = F.cross_entropy(node_probs.unsqueeze(0), 
                                         torch.tensor([target_node_type], device=self.device))
            else:
                node_loss = torch.tensor(0.0, device=self.device)
            
            # Edge loss
            edge_loss = torch.tensor(0.0, device=self.device)
            if target_node < original_graph.x.shape[0]:
                # Get edges connected to the target node in original graph
                target_edges = (original_graph.edge_index[0] == target_node) | \
                             (original_graph.edge_index[1] == target_node)
                
                if target_edges.sum() > 0:
                    target_edge_types = original_graph.edge_attr[target_edges]
                    # For simplicity, use the first edge type as target
                    if len(target_edge_types) > 0:
                        target_edge_type = target_edge_types[0].long().item()
                        edge_loss = F.cross_entropy(edge_probs.unsqueeze(0), 
                                                 torch.tensor([target_edge_type], device=self.device))
            
            total_loss += node_loss + edge_loss
        
        return total_loss / max(len(trajectory) - 1, 1)
    
    def compute_ordering_loss(self, trajectory, node_order, ordering_probs):
        """
        Compute the ordering loss using REINFORCE algorithm.
        
        Args:
            trajectory: List of graphs in diffusion process
            node_order: Order of nodes being absorbed
            ordering_probs: Probabilities used for ordering
            
        Returns:
            loss: Ordering loss
        """
        # Compute reward as negative denoising loss
        denoising_loss = self.compute_denoising_loss(trajectory, node_order, ordering_probs)
        reward = -denoising_loss.detach()
        
        # Compute log probability of the trajectory
        log_prob = 0.0
        for t, (probs, node_idx) in enumerate(zip(ordering_probs, node_order)):
            if node_idx < probs.shape[0]:
                log_prob += torch.log(probs[node_idx] + 1e-8)
        
        # REINFORCE loss
        ordering_loss = -reward * log_prob
        
        return ordering_loss
    
    def train_step(self, train_batch, val_batch=None):
        """
        Perform one training step.
        
        Args:
            train_batch: Batch of training graphs
            val_batch: Batch of validation graphs (optional)
            
        Returns:
            train_loss: Training loss
            val_loss: Validation loss (if val_batch provided)
        """
        self.model.train()
        
        # Training
        total_denoising_loss = 0.0
        total_ordering_loss = 0.0
        num_graphs = 0
        
        for graph in train_batch:
            # Generate M diffusion trajectories
            for _ in range(self.M):
                trajectory, node_order, ordering_probs = self.generate_diffusion_trajectory(graph)
                
                # Compute losses
                denoising_loss = self.compute_denoising_loss(trajectory, node_order, ordering_probs)
                ordering_loss = self.compute_ordering_loss(trajectory, node_order, ordering_probs)
                
                total_denoising_loss += denoising_loss
                total_ordering_loss += ordering_loss
                num_graphs += 1
        
        # Normalize losses
        avg_denoising_loss = total_denoising_loss / num_graphs
        avg_ordering_loss = total_ordering_loss / num_graphs
        
        # Backward pass for denoising network
        self.denoising_optimizer.zero_grad()
        avg_denoising_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.denoising_network.parameters(), max_norm=1.0)
        self.denoising_optimizer.step()
        
        # Backward pass for ordering network
        self.ordering_optimizer.zero_grad()
        avg_ordering_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.diffusion_ordering_network.parameters(), max_norm=1.0)
        self.ordering_optimizer.step()
        
        train_loss = avg_denoising_loss.item() + avg_ordering_loss.item()
        self.train_losses.append(train_loss)
        
        # Logging
        wandb.log({
            "train_denoising_loss": avg_denoising_loss.item(),
            "train_ordering_loss": avg_ordering_loss.item(),
            "train_total_loss": train_loss
        })
        
        # Validation
        val_loss = None
        if val_batch is not None:
            val_loss = self.validate(val_batch)
        
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