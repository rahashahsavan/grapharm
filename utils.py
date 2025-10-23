import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
import numpy as np

def random_node_decay_ordering(datapoint):
    # create random list of nodes
    return torch.randperm(datapoint.x.shape[0]).tolist()

class NodeMasking:
    def __init__(self, dataset):
        self.dataset = dataset
        assert dataset.x.shape[1] == 1, "Only one feature per node is supported"
        
        # Calculate number of unique types
        num_unique_node_types = dataset.x.unique().shape[0]
        num_unique_edge_types = dataset.edge_attr.unique().shape[0]
        
        # Define mask tokens
        self.NODE_MASK = num_unique_node_types  # Mask token for nodes
        self.EMPTY_EDGE = num_unique_edge_types  # Empty edge token
        self.EDGE_MASK = num_unique_edge_types + 1  # Mask token for edges
        
        # Store counts for model initialization
        self.num_node_types = num_unique_node_types + 1  # +1 for MASK token
        self.num_edge_types = num_unique_edge_types + 2  # +2 for EMPTY and MASK tokens
    
    def idxify(self, datapoint):
        """
        Converts node and edge types to indices starting from 0.
        
        CRITICAL: Uses DATASET-WIDE unique types for consistent mapping across all graphs.
        This ensures that NODE_MASK and EDGE_MASK indices are valid for embedding layers.
        
        Args:
            datapoint: PyTorch Geometric Data object
            
        Returns:
            datapoint: Data object with indexed features
        """
        datapoint = datapoint.clone()
        
        # Get unique node and edge types from ENTIRE DATASET (not just this datapoint)
        # This ensures consistent mapping across all graphs
        unique_node_types = self.dataset.x.unique()
        unique_edge_types = self.dataset.edge_attr.unique()
        
        # Create mapping dictionaries
        node_type_to_idx = {node_type.item(): idx for idx, node_type in enumerate(unique_node_types)}
        edge_type_to_idx = {edge_type.item(): idx for idx, edge_type in enumerate(unique_edge_types)}
        
        # Store original device to ensure tensors stay on same device
        device = datapoint.x.device
        
        # Convert node features
        datapoint.x = torch.tensor([node_type_to_idx[node_type.item()] for node_type in datapoint.x], 
                                    device=device).reshape(-1, 1)
        
        # Convert edge attributes
        datapoint.edge_attr = torch.tensor([edge_type_to_idx[edge_type.item()] for edge_type in datapoint.edge_attr],
                                           device=device)
        
        return datapoint
    
    def deidxify(self, datapoint):
        """
        Converts node and edge indices back to their original types.
        
        Args:
            datapoint: PyTorch Geometric Data object
            
        Returns:
            datapoint: Data object with original feature types
        """
        datapoint = datapoint.clone()
        
        # Get unique node and edge types from original dataset
        unique_node_types = self.dataset.x.unique()
        unique_edge_types = self.dataset.edge_attr.unique()
        
        # Create reverse mapping dictionaries
        idx_to_node_type = {idx: node_type.item() for idx, node_type in enumerate(unique_node_types)}
        idx_to_edge_type = {idx: edge_type.item() for idx, edge_type in enumerate(unique_edge_types)}
        
        # Store original device to ensure tensors stay on same device
        device = datapoint.x.device
        
        # Convert node features back
        datapoint.x = torch.tensor([idx_to_node_type.get(node_idx.item(), self.NODE_MASK) 
                                  for node_idx in datapoint.x], device=device).reshape(-1, 1)
        
        # Convert edge attributes back
        datapoint.edge_attr = torch.tensor([idx_to_edge_type.get(edge_idx.item(), self.EDGE_MASK) 
                                          for edge_idx in datapoint.edge_attr], device=device)
        
        return datapoint

    def is_masked(self, datapoint, node=None):
        """
        Check if a node is masked.
        
        Args:
            datapoint: PyTorch Geometric Data object
            node: Node index to check (if None, returns mask status for all nodes)
            
        Returns:
            Boolean or tensor indicating mask status
        """
        if node is None:
            return datapoint.x.squeeze() == self.NODE_MASK
        return datapoint.x[node].item() == self.NODE_MASK

    def remove_node(self, datapoint, node):
        """
        Remove a node from the graph and all edges connected to it.
        
        Args:
            datapoint: PyTorch Geometric Data object
            node: Index of node to remove
            
        Returns:
            datapoint: Data object with node removed
        """
        assert node < datapoint.x.shape[0], "Node does not exist"
        
        if datapoint.x.shape[0] == 1:
            return datapoint.clone()
        
        datapoint = datapoint.clone()
        
        # Remove node from features
        datapoint.x = torch.cat([datapoint.x[:node], datapoint.x[node+1:]])
        
        # Remove edges connected to the node
        if datapoint.edge_index.shape[1] > 0:
            # Find edges not connected to the removed node
            edge_mask = (datapoint.edge_index[0] != node) & (datapoint.edge_index[1] != node)
            
            datapoint.edge_index = datapoint.edge_index[:, edge_mask]
            datapoint.edge_attr = datapoint.edge_attr[edge_mask]
            
            # Update edge indices
            datapoint.edge_index[datapoint.edge_index > node] -= 1
        
        return datapoint

    def add_masked_node(self, datapoint):
        """
        Add a masked node to the graph while preserving existing edges.
        
        CORRECTED: Only edges involving the new masked node are EDGE_MASK.
        Existing edges between unmasked nodes are preserved.
        
        Args:
            datapoint: PyTorch Geometric Data object
            
        Returns:
            datapoint: Data object with masked node added at the END
        """
        datapoint = datapoint.clone()
        n_nodes = datapoint.x.shape[0]
        
        # Store original device to ensure tensors stay on same device
        device = datapoint.x.device
        
        # Add masked node to features (at the end)
        masked_node_feature = torch.tensor([[self.NODE_MASK]], dtype=datapoint.x.dtype, device=device)
        datapoint.x = torch.cat([datapoint.x, masked_node_feature], dim=0)
        
        # Create fully connected graph with (n_nodes + 1) nodes
        # Preserve existing edges, only mask edges involving the new node
        new_edges = []
        new_edge_attrs = []
        
        for i in range(n_nodes + 1):
            for j in range(n_nodes + 1):
                new_edges.append([i, j])
                
                # If edge involves the new masked node (index n_nodes), use EDGE_MASK
                if i == n_nodes or j == n_nodes:
                    new_edge_attrs.append(self.EDGE_MASK)
                else:
                    # Preserve existing edge attribute
                    # Find edge (i, j) in original graph
                    if datapoint.edge_index.shape[1] > 0:
                        edge_mask = (datapoint.edge_index[0] == i) & (datapoint.edge_index[1] == j)
                        if edge_mask.sum() > 0:
                            new_edge_attrs.append(datapoint.edge_attr[edge_mask][0].item())
                        else:
                            new_edge_attrs.append(self.EMPTY_EDGE)
                    else:
                        new_edge_attrs.append(self.EMPTY_EDGE)
        
        # Update edge index and attributes
        datapoint.edge_index = torch.tensor(new_edges, dtype=torch.long, device=device).T
        datapoint.edge_attr = torch.tensor(new_edge_attrs, dtype=datapoint.edge_attr.dtype, device=device)
        
        return datapoint


    def mask_node(self, datapoint, selected_node):
        """
        Mask a node and connect it to ALL other nodes with masked edges.
        
        CORRECTED per user specification:
        - The node itself becomes a [MASK] token
        - Remove original edges of this node
        - Connect the masked node to ALL other nodes with EDGE_MASK
        
        Args:
            datapoint: PyTorch Geometric Data object
            selected_node: Index of node to mask
            
        Returns:
            datapoint: Data object with node masked and connected to all with EDGE_MASK
        """
        datapoint = datapoint.clone()
        n_nodes = datapoint.x.shape[0]
        device = datapoint.x.device
        
        # Step 1: Mask the node
        datapoint.x[selected_node] = self.NODE_MASK
        
        # Step 2: Remove ALL edges connected to this node
        edge_mask = (datapoint.edge_index[0] != selected_node) & \
                    (datapoint.edge_index[1] != selected_node)
        datapoint.edge_index = datapoint.edge_index[:, edge_mask]
        datapoint.edge_attr = datapoint.edge_attr[edge_mask]
        
        # Step 3: Add MASKED edges from selected_node to ALL other nodes
        new_edges_src = []
        new_edges_dst = []
        new_edges_attr = []
        
        for other_node in range(n_nodes):
            if other_node != selected_node:
                # Bidirectional MASK edges
                new_edges_src.extend([selected_node, other_node])
                new_edges_dst.extend([other_node, selected_node])
                new_edges_attr.extend([self.EDGE_MASK, self.EDGE_MASK])
        
        # Add new edges
        if len(new_edges_src) > 0:
            new_edge_index = torch.tensor([new_edges_src, new_edges_dst], 
                                         dtype=torch.long, device=device)
            new_edge_attr = torch.tensor(new_edges_attr, 
                                        dtype=datapoint.edge_attr.dtype, device=device)
            
            datapoint.edge_index = torch.cat([datapoint.edge_index, new_edge_index], dim=1)
            datapoint.edge_attr = torch.cat([datapoint.edge_attr, new_edge_attr], dim=0)
        
        return datapoint
    
    def _reorder_edge_attr_and_index(self, graph):
        '''
        Reorders edge_attr and edge_index to be like on nx graph
        (0, 0), (0, 1), (0, 2), ..., (0, n), (1, 0), (1, 1), ..., (n, n)
        '''
        graph = graph.clone()
        device = graph.x.device
        
        # reorder edge_attr
        edge_attr = torch.full((graph.x.shape[0], graph.x.shape[0]), self.EMPTY_EDGE, dtype=torch.long, device=device)
        for edge_attr_value, edge_index in zip(graph.edge_attr, graph.edge_index.T):
            edge_attr[edge_index[0], edge_index[1]] = edge_attr_value
        graph.edge_attr = edge_attr.view(-1)
        
        # reorder edge_index
        edge_index = torch.stack([torch.tensor([i, j], device=device) for i in range(graph.x.shape[0]) for j in range(graph.x.shape[0])], dim=1)
        graph.edge_index = edge_index.long()
        return graph


    def remove_empty_edges(self, graph):
        '''
        Removes empty edges from graph
        '''
        graph = graph.clone()
        # remove masker.EMPTY_EDGE from edge_attr, and equivalent in edge_index
        graph.edge_index = graph.edge_index[:, graph.edge_attr.squeeze() != self.EMPTY_EDGE]
        graph.edge_attr = graph.edge_attr[graph.edge_attr.squeeze() != self.EMPTY_EDGE]

        return graph

    def demask_node(self, graph, selected_node, node_type, connections_types):
        """
        Demask a node and set its connections to PREVIOUSLY UNMASKED nodes only.
        
        CRITICAL: In reverse denoising (molecule generation), we only predict edges
        to nodes that have already been unmasked. This is different from forward
        diffusion where all nodes are connected.
        
        Args:
            graph: PyTorch Geometric Data object
            selected_node: Index of node to demask
            node_type: Type of the node
            connections_types: Edge types to UNMASKED nodes ONLY (shape: [M])
                              where M = number of currently unmasked nodes (excluding selected_node)
            
        Returns:
            graph: Data object with node demasked and edges set
        """
        graph = graph.clone()
        
        # Get list of unmasked nodes (excluding selected_node)
        unmasked_nodes = [i for i in range(graph.x.shape[0]) 
                          if i != selected_node and not self.is_masked(graph, node=i)]
        
        # CORRECTED ASSERTION: connections_types only for unmasked nodes
        assert connections_types.shape[0] == len(unmasked_nodes), \
            f"connections_types must match number of unmasked nodes: {len(unmasked_nodes)} " \
            f"(got {connections_types.shape[0]}). This is for reverse denoising where we only " \
            f"predict edges to previously unmasked nodes."
        
        # Demask the node
        graph.x[selected_node] = node_type
        
        # Demask edges - only to unmasked nodes
        for idx, connection in enumerate(connections_types):
            i = unmasked_nodes[idx]  # Actual node index in graph
            
            # Find edges between node i and selected_node
            edge_mask = ((graph.edge_index[0] == i) & (graph.edge_index[1] == selected_node)) | \
                       ((graph.edge_index[1] == i) & (graph.edge_index[0] == selected_node))
            graph.edge_attr[edge_mask] = connection
        
        return graph
    def fully_connect(self, graph, keep_original_edges=True):
        """
        Fully connect the graph with edge attributes.
        
        Args:
            graph: PyTorch Geometric Data object
            keep_original_edges: Whether to preserve original edge attributes
            
        Returns:
            graph: Fully connected graph
        """
        graph = graph.clone()
        n_nodes = graph.x.shape[0]
        device = graph.x.device
        
        # Create fully connected adjacency matrix
        edge_index = torch.tensor([(i, j) for i in range(n_nodes) for j in range(n_nodes)], 
                                dtype=torch.long, device=device).T
        
        # Initialize all edges as empty
        edge_attr = torch.full((n_nodes * n_nodes,), self.EMPTY_EDGE, dtype=torch.long, device=device)
        
        if keep_original_edges:
            # Restore original edge attributes
            for i in range(graph.edge_index.shape[1]):
                src, dst = graph.edge_index[:, i]
                edge_idx = src * n_nodes + dst
                edge_attr[edge_idx] = graph.edge_attr[i]
                # Ensure symmetry
                edge_idx_sym = dst * n_nodes + src
                edge_attr[edge_idx_sym] = graph.edge_attr[i]
        
        graph.edge_index = edge_index
        graph.edge_attr = edge_attr
        
        return graph
    
    def generate_fully_masked(self, n_nodes, device='cpu'):
        """
        Generate a fully masked graph.
        
        Args:
            n_nodes: Number of nodes in the graph
            device: Device to create tensors on (default: 'cpu')
            
        Returns:
            graph: Fully masked graph
        """
        # Create node features (all masked)
        x = torch.full((n_nodes, 1), self.NODE_MASK, dtype=torch.long, device=device)
        
        # Create fully connected edge index
        edge_index = torch.tensor([(i, j) for i in range(n_nodes) for j in range(n_nodes)], 
                                dtype=torch.long, device=device).T
        
        # Create edge attributes (all masked)
        edge_attr = torch.full((n_nodes * n_nodes,), self.EDGE_MASK, dtype=torch.long, device=device)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def remove_empty_edges(self, graph):
        """
        Remove empty edges from the graph.
        
        Args:
            graph: PyTorch Geometric Data object
            
        Returns:
            graph: Graph with empty edges removed
        """
        graph = graph.clone()
        
        # Find non-empty edges
        non_empty_mask = graph.edge_attr != self.EMPTY_EDGE
        
        # Remove empty edges
        graph.edge_index = graph.edge_index[:, non_empty_mask]
        graph.edge_attr = graph.edge_attr[non_empty_mask]
        
        return graph
    
    def get_denoised_nodes(self, graph):
        """
        Get list of denoised (non-masked) nodes.
        
        Args:
            graph: PyTorch Geometric Data object
            
        Returns:
            List of denoised node indices
        """
        denoised_nodes = []
        for node in range(graph.x.shape[0]):
            if not self.is_masked(graph, node):
                denoised_nodes.append(node)
        return denoised_nodes
    
    def remove_masked_nodes_and_edges(self, graph):
        """
        Remove all masked nodes and masked edges from the graph.
        
        CRITICAL: This is used when passing trajectory graphs to denoising network.
        In forward diffusion, we keep masked nodes to maintain consistent indexing.
        But when feeding to denoising network, we remove them to get clean subgraph.
        
        Args:
            graph: PyTorch Geometric Data object with possibly masked nodes/edges
            
        Returns:
            cleaned_graph: Graph with all masked nodes/edges removed
            original_to_new_idx: Mapping from original node indices to new indices
        """
        graph = graph.clone()
        device = graph.x.device
        
        # Find unmasked nodes
        unmasked_mask = graph.x.squeeze() != self.NODE_MASK
        unmasked_indices = torch.where(unmasked_mask)[0]
        
        if len(unmasked_indices) == 0:
            # All nodes are masked - return empty graph
            return Data(
                x=torch.empty((0, 1), dtype=graph.x.dtype, device=device),
                edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
                edge_attr=torch.empty((0,), dtype=graph.edge_attr.dtype, device=device)
            ), {}
        
        # Create mapping from original indices to new indices
        original_to_new_idx = {}
        for new_idx, orig_idx in enumerate(unmasked_indices.tolist()):
            original_to_new_idx[orig_idx] = new_idx
        
        # Extract unmasked nodes
        new_x = graph.x[unmasked_indices]
        
        # Filter edges: keep only edges between unmasked nodes and non-MASK edge types
        edge_mask = torch.ones(graph.edge_index.shape[1], dtype=torch.bool, device=device)
        
        for i in range(graph.edge_index.shape[1]):
            src, dst = graph.edge_index[0, i].item(), graph.edge_index[1, i].item()
            edge_type = graph.edge_attr[i].item()
            
            # Keep edge only if both nodes are unmasked and edge is not MASK/EMPTY
            if (src not in original_to_new_idx or 
                dst not in original_to_new_idx or 
                edge_type == self.EDGE_MASK or 
                edge_type == self.EMPTY_EDGE):
                edge_mask[i] = False
        
        # Filter edges
        filtered_edge_index = graph.edge_index[:, edge_mask]
        filtered_edge_attr = graph.edge_attr[edge_mask]
        
        # Remap edge indices to new node indices
        new_edge_index = torch.zeros_like(filtered_edge_index)
        for i in range(filtered_edge_index.shape[1]):
            src = filtered_edge_index[0, i].item()
            dst = filtered_edge_index[1, i].item()
            new_edge_index[0, i] = original_to_new_idx[src]
            new_edge_index[1, i] = original_to_new_idx[dst]
        
        cleaned_graph = Data(
            x=new_x,
            edge_index=new_edge_index,
            edge_attr=filtered_edge_attr
        )
        
        return cleaned_graph, original_to_new_idx
    
    def prepare_denoising_input(self, graph_t_plus_1, target_node_original, previous_nodes_original):
        """
        Prepare input for denoising network at training step t.
        
        CRITICAL SPECIFICATION (from paper):
        At step t (backward: G_{t+1} → G_t):
        - Input: G_{t+1} from forward trajectory
        - Target: σ_t (node masked at step t)
        - Previous: σ(>t) = {σ_{t+1}, ..., σ_n} (nodes after t in ordering)
        
        Input graph G'_{t+1} should have:
        1. Target node σ_t with MASK type
        2. Previous nodes {σ_{t+1}, ..., σ_n} UNMASKED with original types
        3. Original edges between previous nodes
        4. MASKED edges from target to all previous nodes
        
        Args:
            graph_t_plus_1: G_{t+1} from forward trajectory (has multiple masked nodes)
            target_node_original: σ_t (original node index in G_0)
            previous_nodes_original: [σ_{t+1}, ..., σ_n] (original indices)
            
        Returns:
            prepared_graph: G'_{t+1} with only target masked + previous unmasked
            target_idx_new: Index of target node in prepared graph (always 0)
            previous_indices_new: Indices of previous nodes in prepared graph [1, 2, ..., M]
        """
        device = graph_t_plus_1.x.device
        
        # If no previous nodes, return graph with just target masked node
        if len(previous_nodes_original) == 0:
            prepared_graph = Data(
                x=torch.tensor([[self.NODE_MASK]], dtype=torch.long, device=device),
                edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
                edge_attr=torch.empty((0,), dtype=torch.long, device=device)
            )
            return prepared_graph, 0, []
        
        # Step 1: Extract previous nodes (unmasked) and their edges from original graph
        # These nodes should have their ORIGINAL types from G_0, not masked types
        
        # CRITICAL FIX: Validate indices are within bounds
        n_nodes = graph_t_plus_1.x.shape[0]
        valid_previous_nodes = [n for n in previous_nodes_original if n < n_nodes]
        
        if len(valid_previous_nodes) == 0:
            # No valid previous nodes - return only target
            prepared_graph = Data(
                x=torch.tensor([[self.NODE_MASK]], dtype=torch.long, device=device),
                edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
                edge_attr=torch.empty((0,), dtype=torch.long, device=device)
            )
            return prepared_graph, 0, []
        
        # Convert list to tensor for proper CUDA indexing
        previous_nodes_tensor = torch.tensor(valid_previous_nodes, dtype=torch.long, device=device)
        
        previous_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        previous_mask[previous_nodes_tensor] = True
        
        # Get node features for previous nodes
        previous_x = graph_t_plus_1.x[previous_nodes_tensor]
        
        # Get edges between previous nodes
        edge_src_in_prev = previous_mask[graph_t_plus_1.edge_index[0]]
        edge_dst_in_prev = previous_mask[graph_t_plus_1.edge_index[1]]
        edges_between_prev = edge_src_in_prev & edge_dst_in_prev
        
        prev_edge_index = graph_t_plus_1.edge_index[:, edges_between_prev]
        prev_edge_attr = graph_t_plus_1.edge_attr[edges_between_prev]
        
        # Create mapping from original indices to new indices (1-based, 0 is target)
        old_to_new = {}
        # Use the VALID list for mapping (not tensor)
        for new_idx, old_idx in enumerate(valid_previous_nodes, start=1):
            old_to_new[old_idx] = new_idx
        
        # Remap edge indices
        new_edge_index = torch.zeros_like(prev_edge_index)
        for i in range(prev_edge_index.shape[1]):
            src_old = prev_edge_index[0, i].item()
            dst_old = prev_edge_index[1, i].item()
            new_edge_index[0, i] = old_to_new[src_old]
            new_edge_index[1, i] = old_to_new[dst_old]
        
        # Step 2: Add target node at index 0
        target_x = torch.tensor([[self.NODE_MASK]], dtype=torch.long, device=device)
        combined_x = torch.cat([target_x, previous_x], dim=0)
        
        # Step 3: Add masked edges from target (0) to all previous nodes (1, 2, ..., M)
        M = len(valid_previous_nodes)  # Use VALID nodes count
        target_edges_src = []
        target_edges_dst = []
        target_edges_attr = []
        
        for prev_idx_new in range(1, M + 1):
            # Bidirectional masked edges
            target_edges_src.extend([0, prev_idx_new])
            target_edges_dst.extend([prev_idx_new, 0])
            target_edges_attr.extend([self.EDGE_MASK, self.EDGE_MASK])
        
        target_edge_index = torch.tensor([target_edges_src, target_edges_dst], 
                                        dtype=torch.long, device=device)
        target_edge_attr = torch.tensor(target_edges_attr, dtype=torch.long, device=device)
        
        # Combine all edges
        combined_edge_index = torch.cat([target_edge_index, new_edge_index], dim=1)
        combined_edge_attr = torch.cat([target_edge_attr, prev_edge_attr], dim=0)
        
        # Create prepared graph
        prepared_graph = Data(
            x=combined_x,
            edge_index=combined_edge_index,
            edge_attr=combined_edge_attr
        )
        
        # Target is at index 0, previous nodes are at indices [1, 2, ..., M]
        target_idx_new = 0
        previous_indices_new = list(range(1, M + 1))
        
        return prepared_graph, target_idx_new, previous_indices_new