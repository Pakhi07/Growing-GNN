import copy
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch_geometric.data import Data
import torch.nn as nn


class Graph:
    def __init__(self, nodes, edge_dict, labels):
        self.nodes = nodes
        self.edge_dict = edge_dict
        self.labels = labels


    def add_daughter_cells(self, daughters, parent_index, daughter_labels):
        """Add daughter cells in place of the parent cell and update the labels."""
        # print(f"Parent Index: {parent_index}")

        # Step 1: Remove the parent label and get its position
        keys = list(self.labels.keys())
        # print("keys:", keys)
        parent_pos = keys.index(parent_index)

        # Backup the part of the labels before the parent
        new_labels = {k: self.labels[k] for k in keys[:parent_pos]}
        # print("nodes size",self.nodes.size(0))
        # print("before parent:", new_labels)

        # print("parent_pos:", parent_pos)

        # Step 2: Assign the daughter labels at the correct positions
        new_labels[parent_index] = daughter_labels[0]  # First daughter
        new_labels[self.nodes.size(0)] = daughter_labels[1]  # Second daughter

        # print("adding daughters:", new_labels)

        # Step 3: Append the remaining labels after the parent
        new_labels.update({k: self.labels[k] for k in keys[parent_pos + 1:]})

        # Update labels
        self.labels = new_labels
        # print("Updated labels:", self.labels)

        # Step 4: Update the edge dictionary
        self.update_edge_dict(parent_index, daughter_labels[0], daughter_labels[1])

        # Step 4: Add or remove nodes accordingly
        return self.add_remove_nodes(daughters, parent_pos)


    def add_remove_nodes(self, new_nodes, parent_index):
        """Remove the parent node and add new daughter nodes."""
        if new_nodes.dim() == 1:
            new_nodes = new_nodes.unsqueeze(0) 
        
        # Split the array at the parent index into two arrays
        left_nodes = self.nodes[:parent_index]
        right_nodes = self.nodes[parent_index + 1:]  # remove the parent node

        # print("left nodes: ", left_nodes)
        # print("right nodes: ", right_nodes)
        
        # Concatenate the arrays with new nodes in between
        self.nodes = torch.cat([left_nodes, new_nodes, right_nodes])
        # print("Updated nodes:", self.nodes)
        return Graph(self.nodes, self.edge_dict, self.labels), self.nodes


    def update_edge_dict(self, parent_index, daughter1, daughter2):
        """
        Update the edge_dict of the graph after adding daughter cells and removing a parent cell.
        Ensures edges are created between daughters and the parent's neighbors.
        """
        # List of neighbors to reconnect
        parent_neighbors = self.edge_dict.get(parent_index, [])

        # Remove the parent node from the edge dictionary
        if parent_index in self.edge_dict:
            del self.edge_dict[parent_index]

        # Get the indices of the new daughter nodes
        daughter1_index = list(self.labels.keys())[list(self.labels.values()).index(daughter1)]
        daughter2_index = list(self.labels.keys())[list(self.labels.values()).index(daughter2)]

        print(f"parent: {parent_index}, daughters: {daughter1_index}, {daughter2_index}")
        
        # Add new entries for daughter nodes
        self.edge_dict[daughter1_index] = [daughter2_index] + parent_neighbors  # d -> [e, b, c]
        self.edge_dict[daughter2_index] = [daughter1_index] + parent_neighbors  # e -> [d, b, c]

        print(f"D1: {self.edge_dict[daughter1_index]}")
        print(f"D2: {self.edge_dict[daughter2_index]}")

        # Reconnect edges that were pointing to the parent to the new daughters
        for node, connections in self.edge_dict.items():
            if parent_index in connections:
                connections.remove(parent_index)
                if node not in [daughter1_index, daughter2_index]:
                    connections.extend([daughter1_index, daughter2_index])
            print(f"Node: {node}, Connections: {connections}")



    def to_data(self):
        """Convert the graph to PyTorch Geometric Data object with edge weights."""
        edges = []
        edge_weights = []

        # Iterate over the edge dictionary to extract edges and compute weights
        for node, destinations in self.edge_dict.items():
            for d in destinations:
                edges.append([node, d])

                # Extract positions from node embeddings
                pos1 = self.get_node_position_from_embeddings(node)
                pos2 = self.get_node_position_from_embeddings(d)
                weight = self.euclidean_distance_3d(pos1, pos2)
                edge_weights.append(weight)

        # Convert to tensor format for PyTorch Geometric
        edges = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.nodes.device)
        edge_weights = torch.tensor(edge_weights, dtype=torch.float).to(self.nodes.device)

        # Create a PyTorch Geometric Data object
        return Data(
            x=self.nodes,
            edge_index=edges,
            edge_attr=edge_weights,  # Include edge weights
        )


    def get_node_position_from_embeddings(self, node_index):
        """Helper method to extract the position of a node from its embeddings."""
        # Assuming the first 3 dimensions of self.nodes represent x, y, z coordinates
        # print("node_iembeddings",self.nodes[node_index][:3])
        return self.nodes[node_index][:3]  # Extract (x, y, z) from the embeddings


    def euclidean_distance_3d(self, pos1, pos2):
        """Calculate Euclidean distance between two nodes in 3D space."""
        return torch.sqrt(torch.sum((pos1 - pos2) ** 2))


    # def to_data(self):
    #     """Convert the graph to PyTorch Geometric Data object."""
    #     edges = []
    #     for node in self.edge_dict:
    #         destinations = self.edge_dict[node]
    #         for d in destinations:
    #             edges.append([node, d])
    # 
    #     edges = torch.tensor(edges).long().t().contiguous().to(self.nodes.device)
    #     return Data(
    #         x=self.nodes * torch.ones(self.nodes.size(), device=self.nodes.device),
    #         edge_index=edges,
    #     )

    def plot(self, fig=None, node_colors=None):
        """Plot the graph using NetworkX and Matplotlib."""
        data = self.to_data()
        G = torch_geometric.utils.to_networkx(data, to_undirected=True)
        # print(f"Edge Index: {data.edge_index}")
        # print(f"Number of Edges: {data.edge_index.size(1)}")  # Number of edges
        #print position from node embeddings
        # print(f"Node Embeddings: {self.nodes[:3]}")
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")

        if pos is None:
            print("Positioning failed with Graphviz. Check for issues in graph structure.")
        # else:
        #     print(f"Positions: {pos}")

        if fig is None:
            fig = plt.figure()
        canvas = FigureCanvas(fig)

        edgelist = [(key, value) for key, values in self.edge_dict.items() for value in values]
        print("edgelist",edgelist)

        nx.draw_networkx_nodes(G, pos, node_color = node_colors)
        nx.draw_networkx_edges(G, pos, edgelist = edgelist, edge_color="black")
        nx.draw_networkx_labels(G, pos, labels=self.labels)

        canvas.draw()  # draw the canvas, cache the renderer

        image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image

    def copy(self):
        """Create a deep copy of the graph."""
        nodes = self.nodes * torch.ones(self.nodes.size(), device=self.nodes.device)
        edge_dict = copy.deepcopy(self.edge_dict)
        labels = copy.deepcopy(self.labels)
        return Graph(nodes, edge_dict, labels)
