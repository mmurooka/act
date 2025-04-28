import torch
import torch.nn as nn

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, edge_index):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

        # Create adjacency matrix from edge_index
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

        # edge_index: (2, num_edges)
        adj[edge_index[0], edge_index[1]] = 1.0

        # Add self-loops
        adj += torch.eye(num_nodes)

        # Degree matrix
        deg = adj.sum(dim=1)  # (num_nodes,)

        # D^{-1/2}
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0  # avoid nan

        # Normalized adjacency: D^{-1/2} A D^{-1/2}
        adj_normalized = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)

        # Save normalized adjacency
        self.register_buffer('adj_norm', adj_normalized)

    def forward(self, x):
        # x: [batch_size, num_nodes, in_features]
        batch_size, num_nodes, in_features = x.shape

        adj = self.adj_norm.unsqueeze(0).expand(batch_size, -1, -1)  # Broadcast to batch

        x = torch.bmm(adj, x)  # [batch_size, num_nodes, in_features]
        x = self.fc(x)         # Linear layer
        return x
