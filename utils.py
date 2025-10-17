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
        
        # Define mask tokens
        self.NODE_MASK = dataset.x.unique().shape[0]  # Mask token for nodes
        self.EMPTY_EDGE = dataset.edge_attr.unique().shape[0]  # Empty edge token
        self.EDGE_MASK = dataset.edge_attr.unique().shape[0] + 1  # Mask token for edges
    
    def idxify(self, datapoint):
        """
        Converts node and edge types to indices starting from 0.
        
        Args:
            datapoint: PyTorch Geometric Data object
            
        Returns:
            datapoint: Data object with indexed features
        """
        datapoint = datapoint.clone()
        
        # Get unique node and edge types
        unique_node_types = datapoint.x.unique()
        unique_edge_types = datapoint.edge_attr.unique()
        
        # Create mapping dictionaries
        node_type_to_idx = {node_type.item(): idx for idx, node_type in enumerate(unique_node_types)}
        edge_type_to_idx = {edge_type.item(): idx for idx, edge_type in enumerate(unique_edge_types)}
        
        # Convert node features
        datapoint.x = torch.tensor([node_type_to_idx[node_type.item()] for node_type in datapoint.x]).reshape(-1, 1)
        
        # Convert edge attributes
        datapoint.edge_attr = torch.tensor([edge_type_to_idx[edge_type.item()] for edge_type in datapoint.edge_attr])
        
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
        
        # Convert node features back
        datapoint.x = torch.tensor([idx_to_node_type.get(node_idx.item(), self.NODE_MASK) 
                                  for node_idx in datapoint.x]).reshape(-1, 1)
        
        # Convert edge attributes back
        datapoint.edge_attr = torch.tensor([idx_to_edge_type.get(edge_idx.item(), self.EDGE_MASK) 
                                          for edge_idx in datapoint.edge_attr])
        
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
        Add a masked node to the graph.
        
        Args:
            datapoint: PyTorch Geometric Data object
            
        Returns:
            datapoint: Data object with masked node added
        """
        datapoint = datapoint.clone()
        n_nodes = datapoint.x.shape[0]
        
        # Add masked node to features
        masked_node_feature = torch.tensor([[self.NODE_MASK]], dtype=datapoint.x.dtype)
        datapoint.x = torch.cat([datapoint.x, masked_node_feature], dim=0)
        
        # Create new edges connecting the masked node to all existing nodes
        new_edges = []
        new_edge_attrs = []
        
        for i in range(n_nodes + 1):
            for j in range(n_nodes + 1):
                new_edges.append([i, j])
                new_edge_attrs.append(self.EDGE_MASK)
        
        # Update edge index and attributes
        datapoint.edge_index = torch.tensor(new_edges, dtype=torch.long).T
        datapoint.edge_attr = torch.tensor(new_edge_attrs, dtype=datapoint.edge_attr.dtype)
        
        return datapoint


    def mask_node(self, datapoint, selected_node):
        """
        Mask a node and all its connecting edges.
        
        Args:
            datapoint: PyTorch Geometric Data object
            selected_node: Index of node to mask
            
        Returns:
            datapoint: Data object with node masked
        """
        datapoint = datapoint.clone()
        
        # Mask the node
        datapoint.x[selected_node] = self.NODE_MASK
        
        # Mask all edges connected to this node
        edge_mask = (datapoint.edge_index[0] == selected_node) | (datapoint.edge_index[1] == selected_node)
        datapoint.edge_attr[edge_mask] = self.EDGE_MASK
        
        return datapoint
    
    def _reorder_edge_attr_and_index(self, graph):
        '''
        Reorders edge_attr and edge_index to be like on nx graph
        (0, 0), (0, 1), (0, 2), ..., (0, n), (1, 0), (1, 1), ..., (n, n)
        '''
        graph = graph.clone()
        # reorder edge_attr
        edge_attr = torch.full((graph.x.shape[0], graph.x.shape[0]), self.EMPTY_EDGE, dtype=torch.long)
        for edge_attr_value, edge_index in zip(graph.edge_attr, graph.edge_index.T):
            edge_attr[edge_index[0], edge_index[1]] = edge_attr_value
        graph.edge_attr = edge_attr.view(-1)
        
        # reorder edge_index
        edge_index = torch.stack([torch.tensor([i, j]) for i in range(graph.x.shape[0]) for j in range(graph.x.shape[0])], dim=1)
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
        Demask a node and set its connections.
        
        Args:
            graph: PyTorch Geometric Data object
            selected_node: Index of node to demask
            node_type: Type of the node
            connections_types: Edge types for connections
            
        Returns:
            graph: Data object with node demasked
        """
        assert connections_types.shape[0] == graph.x.shape[0], "Number of connections must equal number of nodes"
        
        graph = graph.clone()
        
        # Demask the node
        graph.x[selected_node] = node_type
        
        # Demask edges
        for i, connection in enumerate(connections_types):
            if not self.is_masked(graph, node=i):
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
        
        # Create fully connected adjacency matrix
        edge_index = torch.tensor([(i, j) for i in range(n_nodes) for j in range(n_nodes)], 
                                dtype=torch.long).T
        
        # Initialize all edges as empty
        edge_attr = torch.full((n_nodes * n_nodes,), self.EMPTY_EDGE, dtype=torch.long)
        
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
    
    def generate_fully_masked(self, n_nodes):
        """
        Generate a fully masked graph.
        
        Args:
            n_nodes: Number of nodes in the graph
            
        Returns:
            graph: Fully masked graph
        """
        # Create node features (all masked)
        x = torch.full((n_nodes, 1), self.NODE_MASK, dtype=torch.long)
        
        # Create fully connected edge index
        edge_index = torch.tensor([(i, j) for i in range(n_nodes) for j in range(n_nodes)], 
                                dtype=torch.long).T
        
        # Create edge attributes (all masked)
        edge_attr = torch.full((n_nodes * n_nodes,), self.EDGE_MASK, dtype=torch.long)
        
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