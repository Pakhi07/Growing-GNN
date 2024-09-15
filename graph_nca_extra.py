import random
from typing import Optional
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.optim as optim
import cma


# from growing_nn.graph.directed_graph import DirectedGraph
csv_path = '/home/pakhi/Documents/gsoc/gsoc-2024/Growing-GNNs/developmental-cells-and-position.csv'


class GraphNCA(nn.Module):
    def __init__(self, graph, num_hidden_channels: int = 16, max_replications: int = 2):
        super().__init__()
        self.graph = graph


        self.predicted_distance_matrix = None
        self.true_distance_matrix = None



        # self.positions = torch.rand(nodes.size(0), 3, requires_grad=True)  # Random initial positions
        self.true_positions = self.load_positions_from_csv(csv_path)
        
        # self.num_input_nodes = self.graph.num_input_nodes
        # self.num_output_nodes = self.graph.num_output_nodes

        # self.input_nodes = self.graph.input_nodes
        # self.output_nodes = self.graph.output_nodes

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

    @classmethod
    def get_number_of_channels(
        cls, num_operations: int, num_activations: int, num_hidden_channels
    ):
        return num_hidden_channels
    


    # def initialize_distance_matrices(self):
    #     """Initialize the distance matrices based on the current number of nodes."""
    #     num_nodes = len(self.graph.labels)
    #     self.predicted_distance_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    #     self.true_distance_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)


    def forward(self, xx, edge_index, parent_index):
        # print("x.shape",xx.shape)
        # print("edge_index.shape",edge_index.shape)
        features = self.perception_net(xx, edge_index)
        update = self.update_net(features)
        # print("grrad check",xx.requires_grad)
        xx = xx.clone() + update
        # print("grrad check2",xx.requires_grad)

        return xx
    

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

    # def calculate_distances(self, daughter_pairs):
    #     """Calculate Euclidean distances between pairs of daughter cells."""
    #     distances = []
    #     data = self.graph.to_data()
    #     # print(daughter_pairs)
    
    #     # for (label1, label2) in daughter_pairs:
    #         # find index for label1 and label2 from graph
    #     idx1 = list(self.graph.labels.keys())[list(self.graph.labels.values()).index(daughter_pairs[0])]
    #     idx2 = list(self.graph.labels.keys())[list(self.graph.labels.values()).index(daughter_pairs[1])]
    #     dist = self.euclidean_distance_3d(data.x[idx1], data.x[idx2])
    #     distances.append(dist)
    #     return torch.stack(distances)

    # def calculate_true_distances(self, daughter_pairs):
    #     """Calculate true Euclidean distances between pairs of daughter cells based on CSV coordinates."""
    #     true_distances = []
    #     # print("true_positions",self.true_positions)
    #     # print("daughter_pairs",daughter_pairs)

    #     # for (label1, label2) in daughter_pairs:
    #     pos1 = self.true_positions[daughter_pairs[0]]
    #     pos2 = self.true_positions[daughter_pairs[1]]
    #     dist = self.euclidean_distance_3d(pos1, pos2)
    #     true_distances.append(dist)
    #     return torch.stack(true_distances)

    # def calculate_loss(self, predicted_distances, true_distances):
    #     """Calculate loss between predicted and true distances."""
    #     criterion = nn.MSELoss()
    #     return criterion(predicted_distances, true_distances)

    # def assign_positions_and_calculate_loss(self, daughter_pairs):
    #     """Assign positions, calculate distances, and compute loss."""
    #     predicted_distances = self.calculate_distances(daughter_pairs)
    #     true_distances = self.calculate_true_distances(daughter_pairs)
    #     loss = self.calculate_loss(predicted_distances, true_distances)
    #     return loss

    # def optimize_positions(self, graph, daughter1, daughter2, updated_nodes, learning_rate=0.01, epochs=100):
    #     """Optimize positions to minimize loss based on true distances."""
    #     self.graph = graph.copy()
    #     torch.autograd.set_detect_anomaly(True)
    #     optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    #     # Create a mapping from daughter labels to their index in the updated nodes
    #     daughter1_index = list(graph.labels.keys())[list(graph.labels.values()).index(daughter1)]
    #     daughter2_index = list(graph.labels.keys())[list(graph.labels.values()).index(daughter2)]
        
    #     # print(f"Daughter 1 Index: {daughter1_index}, Daughter 2 Index: {daughter2_index}")

    #     # Optimization loop
    #     for epoch in range(epochs):
    #         print(f"Epoch {epoch}")
    #         optimizer.zero_grad()

    #         # Forward propagation: Pass the node embeddings through the model's layers
    #         data = self.graph.to_data()
    #         xx = self.forward(data.x, data.edge_index, daughter1_index)  # Update xx through the forward pass

    #         # Calculate Euclidean distances between updated nodes
    #         distance = self.euclidean_distance_3d(xx[daughter1_index], xx[daughter2_index])

    #         # Calculate true Euclidean distance
    #         pos1 = self.true_positions[daughter1]
    #         pos2 = self.true_positions[daughter2]
    #         true_distance = self.euclidean_distance_3d(pos1, pos2)

    #         # Calculate loss
    #         criterion = nn.MSELoss()
    #         loss = criterion(distance, true_distance)
            
    #         # Perform the backward pass
    #         loss.backward(retain_graph = True)  # No retain_graph=True is needed here

    #         # Update model parameters
    #         optimizer.step()

    #         # After backward pass, you can log the loss (but do not use it in computations)
    #         print(f'Epoch {epoch}, Loss: {loss.item()}')
        
    #     return self.graph



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

        # Forward pass to update node features
        # data = self.graph.to_data()
        # xx = self.forward(data.x, data.edge_index, parent_index)
        
        # Update the distance matrices to reflect current graph state
        self.update_distance_matrices()

        # Calculate loss based on the updated distance matrices
        loss = self.calculate_loss()
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

    def optimize_with_cma_es(self, graph, parent_index, daughter_pairs):
        """
        Optimize the model parameters using CMA-ES.
        """
        # Initialize CMA-ES
        initial_params = torch.cat([p.flatten() for p in self.parameters()]).detach().numpy()
        # print("params: ",initial_params)
        es = cma.CMAEvolutionStrategy(initial_params, 0.5)  # Initial mean and std deviation

        self.graph = graph.copy()

        # Optimize
        while not es.stop():
            solutions = es.ask()
            # print("solutions: ",len(solutions))
            # print("solutions[0]: ",solutions[0].shape)
            losses = [self.objective_function(x) for x in solutions]
            # print("losses: ",losses)
            loss_tensor = torch.tensor(losses)
            loss = torch.clamp(loss_tensor, min=-1e6, max=1e6)
            #convert tensor back to list
            loss = loss.tolist()
            # print("losses after clamping: ",loss)
            es.tell(solutions, loss)

            

        # Set the optimized parameters
        optimized_params = es.result.xbest
        self.set_parameters(optimized_params)



    def update_distance_matrices(self):
        """Update distance matrices whenever new cells are formed."""
        data = self.graph.to_data()
        num_nodes = data.x.size(0)

        # Expand the matrices if new cells are added
        if self.predicted_distance_matrix is None or self.predicted_distance_matrix.size(0) != num_nodes:
            self.predicted_distance_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
            self.true_distance_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

        # Update predicted distance matrix
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = self.euclidean_distance_3d(data.x[i], data.x[j])
                self.predicted_distance_matrix[i, j] = dist
                self.predicted_distance_matrix[j, i] = dist

        # Update true distance matrix
        labels = list(self.graph.labels.values())
        print("labels: ", self.graph.labels)
        # print("labels 2:", labels)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                label1 = labels[i]
                label2 = labels[j]
                # print("label1",label1)
                # print("label2",label2)
                pos1 = self.true_positions[label1]
                pos2 = self.true_positions[label2]
                dist = self.euclidean_distance_3d(pos1, pos2)
                self.true_distance_matrix[i, j] = dist
                self.true_distance_matrix[j, i] = dist

        print("Updated predicted distance matrix:\n", self.predicted_distance_matrix)
        print("Updated true distance matrix:\n", self.true_distance_matrix)

    def calculate_loss(self):
        """Calculate the MSE loss between the predicted and true distance matrices."""
        criterion = nn.MSELoss()
        return criterion(self.predicted_distance_matrix, self.true_distance_matrix)




    def grow(
        self,
        graph,
        parent_index,
        daughter_labels,
    ):
        new_graph = graph.copy()
        data = new_graph.to_data()
    

        xx = self.forward(data.x, data.edge_index, parent_index)
        # print("xx grad check grow",xx.requires_grad)

        split = self.split_network(xx[parent_index])
        
        daughter1 = split[:self.num_channels]
        daughter2 = split[self.num_channels:]
        daughters = torch.stack([daughter1, daughter2])


        #add daughter cells to the graph and remove the parent cell from the graph
        new_graph, updated_nodes = new_graph.add_daughter_cells(daughters, parent_index, daughter_labels)   
        # print new added cells 
        # print("new added cells",new_graph.nodes.shape)

        # self.update_distance_matrices(daughter_labels)

        #create edges between the parent cell and the daughter cells and pass old edge disctionary too
        # new_graph = new_graph.add_edges(new_graph.edge_dict, parent_index)
        # print("grown graph",new_graph.nodes.shape)

        return new_graph, updated_nodes