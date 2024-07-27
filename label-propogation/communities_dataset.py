import torch
from torch_geometric.data import Data
import numpy as np
import networkx as nx
import random

class createDataset(torch.utils.data.Dataset):
    def __init__(self, num_nodes, num_classes, q):
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.q = q
        self.p = 1 - q
        self.data_list = []
        self.classes= np.zeros(num_nodes)
    
    def generate_grap(self):
        # Initialize the graph
        G = nx.Graph()
        
        # Add nodes to the graph
        G.add_nodes_from(range(self.num_nodes))
        
        # Assign nodes to classes
        classes = {i: [] for i in range(self.num_classes)}
        nodes = list(range(self.num_nodes))
        random.shuffle(nodes)
    
        for i, node in enumerate(nodes):
            classes[i % self.num_classes].append(node)
    
        # Create edges based on probabilities
        for i in range(self.num_nodes):
    
            for j in range(i + 1, self.num_nodes):
                same_class = any(i in classes[c] and j in classes[c] for c in range(self.num_classes))
                if same_class:
                    if random.random() < self.p:
                        G.add_edge(i, j)
                else:
                    if random.random() < self.q:
                        G.add_edge(i, j)

        self.enter_classes(classes)    
        return G
    

    def enter_classes(self,classes):
        tempArr= [c for c in classes.values()]
        for i in range(len(tempArr)):
            for j in tempArr[i]:

                self.classes[j]= i
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    