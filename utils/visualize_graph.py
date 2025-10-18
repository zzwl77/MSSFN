import torch
import matplotlib.pyplot as plt

def create_adjacency_matrix(edge_index, num_nodes):
    # Create an empty adjacency matrix
    adjacency_matrix = torch.zeros((num_nodes, num_nodes))
    
    # Populate the adjacency matrix
    for edge in edge_index.t():
        src, tgt = edge[0], edge[1]
        adjacency_matrix[src, tgt] = 1  # Mark the presence of an edge

    plt.figure(figsize=(8, 6))
    plt.imshow(adjacency_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Adjacency Matrix Visualization')
    plt.xlabel('Target Nodes')
    plt.ylabel('Source Nodes')
    plt.show()

    return adjacency_matrix