import random
from typing import Optional
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.optim as optim



csv_path = '/home/pakhi/Documents/gsoc/gsoc-2024/Growing-GNNs/developmental-cells-and-position.csv'


class GraphNCA(nn.Module):
    def __init__(self, graph, num_hidden_channels: int = 16, max_replications: int = 2):
        super().__init__()
        self.graph = graph

        self.true_positions = self.load_positions_from_csv(csv_path)


        self.value_idx = 0
        self.replication_idx = 1

        self.operations = [torch.add, torch.subtract, torch.multiply]
        self.activations = [torch.relu, torch.tanh]

        self.replicated_cells = []
        self.num_operations = len(self.operations)
        self.num_activations = len(self.activations)

        self.operation_channels = [2, 4]
        self.activation_channels = [5, 6]

        self.num_hidden_channels = num_hidden_channels
        self.num_channels = self.get_number_of_channels(
            self.num_operations, self.num_activations, self.num_hidden_channels
        )

        self.perception_net = GCNConv(
            self.num_channels, self.num_channels * 3, bias=False
        )
        self.update_net = nn.Sequential(
            nn.Linear(self.num_channels * 3, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_channels),
        )
        self.split_network = nn.Sequential(        
            nn.Linear(self.num_channels, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_channels * 2),
        )
        self.max_replications = max_replications
        self.adjacency_matrix = None

    @classmethod
    def get_number_of_channels(
        cls, num_operations: int, num_activations: int, num_hidden_channels
    ):
        return num_hidden_channels

    def forward(self, xx, edge_index, parent_index):
        print("x.shape",xx.shape)
        print("edge_index.shape",edge_index.shape)
        features = self.perception_net(xx, edge_index)
        update = self.update_net(features)
        print("grrad check",xx.requires_grad)
        xx = xx.clone() + update
        split = self.split_network(xx[parent_index])
        
        daughter1 = split[:self.num_channels]
        daughter2 = split[self.num_channels:]
        daughters = torch.stack([daughter1, daughter2])
        return xx, daughters
    

    def load_positions_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        true_positions = {}
        for _, row in df.iterrows():
            cell_label = row['Parent Cell']
            position = (row['parent_x'], row['parent_y'], row['parent_z'])
            true_positions[cell_label] = torch.tensor(position, dtype=torch.float32)
        return true_positions

    def euclidean_distance_3d(self, node1, node2):
        """Calculate Euclidean distance between two nodes in 3D space."""
        return torch.sqrt(torch.sum((node1 - node2) ** 2))
    

    def initialize_adjacency_matrix(self, num_nodes):
        """Initialize adjacency matrix with zeros."""
        self.adjacency_matrix = torch.zeros((num_nodes, num_nodes))

    def update_adjacency_matrix(self):
        """Update adjacency matrix with Euclidean distances between all nodes."""
        data = self.graph.to_data()
        num_nodes = data.x.size(0)
        self.initialize_adjacency_matrix(num_nodes)
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = self.euclidean_distance_3d(data.x[i], data.x[j])
                self.adjacency_matrix[i, j] = dist
                self.adjacency_matrix[j, i] = dist  # Since the matrix is symmetric

    def calculate_loss_from_adjacency_matrix(self):
        """Calculate loss from the adjacency matrix."""
        if self.adjacency_matrix is None:
            raise ValueError("Adjacency matrix has not been initialized.")
        
        # Example loss calculation (sum of all distances)
        loss = torch.sum(self.adjacency_matrix)
        return loss


    def calculate_distances(self, daughter_pairs):
        """Calculate Euclidean distances between pairs of daughter cells."""
        distances = []
        data = self.graph.to_data()
    
        for (label1, label2) in daughter_pairs:
            # find index for label1 and label2 from graph
            idx1 = list(self.graph.labels.keys())[list(self.graph.labels.values()).index(label1)]
            idx2 = list(self.graph.labels.keys())[list(self.graph.labels.values()).index(label2)]
            dist = self.euclidean_distance_3d(data.x[idx1], data.x[idx2])
            distances.append(dist)
        return torch.stack(distances)

    def calculate_true_distances(self, daughter_pairs):
        """Calculate true Euclidean distances between pairs of daughter cells based on CSV coordinates."""
        true_distances = []
        # print("true_positions",self.true_positions)
        print("daughter_pairs",daughter_pairs)

        for (label1, label2) in daughter_pairs:
            pos1 = self.true_positions[label1]
            pos2 = self.true_positions[label2]
            dist = self.euclidean_distance_3d(pos1, pos2)
            true_distances.append(dist)
        return torch.stack(true_distances)


    def assign_positions_and_calculate_loss(self, daughter_pairs):
        """Assign positions, calculate distances, and compute loss."""
        # predicted_distances = self.calculate_distances(daughter_pairs)
        # true_distances = self.calculate_true_distances(daughter_pairs)
        loss = self.calculate_loss_from_adjacency_matrix()
        return loss

    def optimize_positions(self, graph, daughter_pairs, learning_rate=0.01, epochs=100):
        """Optimize positions to minimize loss based on true distances."""
        self.graph = graph.copy()
        # data = self.graph.to_data()
        torch.autograd.set_detect_anomaly(True)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.assign_positions_and_calculate_loss(daughter_pairs)
            loss.backward()
            optimizer.step()
            # update graph node positions

            if epoch % 10 == 0:  # Print loss every 10 epochs
                print(f'Epoch {epoch}, Loss: {loss.item()}')
            
        return self.graph
        

    def grow(
        self,
        graph,
        parent_index,
        daughter_labels,
    ):
        new_graph = graph.copy()
        data = new_graph.to_data()
    

        xx, daughters = self.forward(data.x, data.edge_index, parent_index)
        print("x grad check",xx.requires_grad)


        #add daughter cells to the graph and remove the parent cell from the graph
        new_graph, updated_nodes = new_graph.add_daughter_cells(daughters, parent_index, daughter_labels)

        #create edges between the parent cell and the daughter cells and pass old edge disctionary too
        # new_graph = new_graph.add_edges(new_graph.edge_dict, parent_index)
        print("grown graph",new_graph.nodes.shape)

        return new_graph, updated_nodes
    









    def objective_function(self, params):
        """
        The objective function to minimize. 
        Args:
            params (list): List of network parameters.
        Returns:
            float: Loss value to minimize.
        """
        # Assign the parameters to the model
        self.set_parameters(params)

        # Forward pass and calculate loss
        data = self.graph.to_data()
        xx, _ = self.forward(data.x, data.edge_index, parent_index)
        
        # Define your pairs and calculate distances
        predicted_distances = self.calculate_distances(daughter_pairs)
        true_distances = self.calculate_true_distances(daughter_pairs)
        loss = self.calculate_loss(predicted_distances, true_distances)
        return loss.item()

    def set_parameters(self, params):
        """
        Set the parameters of the model based on the optimization vector.
        Args:
            params (list): List of parameters to set in the model.
        """
        # Convert the list of parameters into tensors and assign them to model weights
        start = 0
        for p in self.parameters():
            end = start + p.numel()
            p.data.copy_(torch.tensor(params[start:end]).view_as(p))
            start = end

    def optimize_with_cma_es(self):
        """
        Optimize the model parameters using CMA-ES.
        """
        # Initialize CMA-ES
        initial_params = torch.cat([p.flatten() for p in self.parameters()]).numpy()
        es = cma.CMAEvolutionStrategy(initial_params, 0.5)  # Initial mean and std deviation

        # Optimize
        while not es.stop():
            solutions = es.ask()
            losses = [self.objective_function(x) for x in solutions]
            es.tell(solutions, losses)

        # Set the optimized parameters
        optimized_params = es.result.xbest
        self.set_parameters(optimized_params)