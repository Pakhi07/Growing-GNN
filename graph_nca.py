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
csv_path = '/home/pakhi/Documents/gsoc/gsoc-2024/Growing-GNNs/merged_common_cells.csv'


class GraphNCA(nn.Module):
    def __init__(self, graph, num_hidden_channels: int = 16, max_replications: int = 2):
        super().__init__()
        self.graph = graph

        self.predicted_distance_matrix = None
        self.true_distance_matrix = None

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

    @classmethod
    def get_number_of_channels(
        cls, num_operations: int, num_activations: int, num_hidden_channels
    ):
        return num_hidden_channels
    

    def forward(self, xx, edge_index, parent_index):
        # Check if initial input has NaN or Inf
        if torch.isnan(xx).any() or torch.isinf(xx).any():
            print(f"NaN/Inf in input node features (xx) before GCN: {xx}")
        
        features = self.perception_net(xx, edge_index)

        # Check after GCN
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"NaN/Inf in node features after GCNConv: {features}")
        
        update = self.update_net(features)

        # Check after update net
        if torch.isnan(update).any() or torch.isinf(update).any():
            print(f"NaN/Inf in node features after update_net: {update}")
        
        xx = xx.clone() + update
        
        # Check final output of forward pass
        if torch.isnan(xx).any() or torch.isinf(xx).any():
            print(f"NaN/Inf in output node features (xx) after update: {xx}")

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
        diff = node1 - node2
        # if torch.isnan(diff).any() or torch.isinf(diff).any():
        #     print(f"NaN/Inf detected in difference between node1 and node2: {node1}, {node2}")

        dist = torch.sqrt(torch.clamp(torch.sum(diff ** 2), min=1e-9, max=1e9))
        # if torch.isnan(dist) or torch.isinf(dist):
        #     print(f"NaN/Inf detected in distance calculation: {dist}")
    
        return dist


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
        # print("params: ",params)
        for p in self.parameters():
            end = start + p.numel()
            # print("p.numel()",p.numel())
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
            loss = torch.clamp(loss_tensor, min=-1e6, max=1e6).tolist()
            # print("losses after clamping: ",loss)
            es.tell(solutions, loss)

        # Set the optimized parameters
        optimized_params = es.result.xbest
        # print("optimized_params: ",optimized_params)
        self.set_parameters(optimized_params)


    def validate_data(self, data):
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            print("Input data contains NaN or inf values!")
            return False
        return True


    def update_distance_matrices(self):
        """Update distance matrices whenever new cells are formed."""
        data = self.graph.to_data()
        data.x = torch.clamp(data.x, min=-1e6, max=1e6)
        data.x = torch.nan_to_num(data.x, nan=0.0, posinf=1e6, neginf=-1e6)


        # if torch.isnan(data.x).any() or torch.isinf(data.x).any():
        #     print("-.-.-.-.-.-.- NaN/Inf in node features (data.x)!")
        # if not self.validate_data(data):
        #     print("Invalid input data detected!")
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
        # print("labels: ", self.graph.labels)
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

        # print("Updated predicted distance matrix:\n", self.predicted_distance_matrix)
        # print("Updated true distance matrix:\n", self.true_distance_matrix)

    def calculate_loss(self):
        """Calculate the MSE loss between the predicted and true distance matrices."""
        # if torch.isnan(self.predicted_distance_matrix).any() or torch.isinf(self.predicted_distance_matrix).any():
        #     print("NaN/Inf in predicted_distance_matrix")
            
        # if torch.isnan(self.true_distance_matrix).any() or torch.isinf(self.true_distance_matrix).any():
        #     print("NaN/Inf in true_distance_matrix")

        criterion = nn.MSELoss()
        mse_loss = criterion(self.predicted_distance_matrix, self.true_distance_matrix)
        # if torch.isnan(mse_loss):
        #     print("NaN in MSE loss!")

        # Add L2 regularization to prevent large weights
        l2_lambda = 1e-5
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        total_loss = mse_loss + l2_lambda * l2_norm

        # if torch.isnan(total_loss):
        #     print("NaN in total loss after L2 regularization!")

        return total_loss




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

        return new_graph, updated_nodes