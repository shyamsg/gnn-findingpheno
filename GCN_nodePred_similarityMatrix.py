from torch_geometric.data import Data
import numpy as np
import pandas as pd
from scipy.spatial import distance
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
from torch_geometric.utils import is_undirected
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, TUDataset # to check the types of dataset variables

from similarity_graph_utilities import get_edges_from_adjacency, plot_feature_correlation


"""
 Implementing a Graph Convolutional Network (GCN) for node classification

 INPUTS:
    - adjacency_matrix: matrix n_samples x n_samples whose values represent similarities.
    - MG_T: matrix n_samples x n_features whose values represent METAGENOMIC and TRANSCRIPTOMICS FEATURES of the samples.
    - Pheno: matrix n_samples x 1 whose values represent the PHENOTYPE of the samples.
    
"""

# Regularization functions that account for connectivity: Use 5 or 10 as minimum nunmber of neighbors

# Things to try: Loss functions (MAE; MSE); Optimizers; Layers



def main():
    
    adjacency_matrix = pd.read_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/output/adjacency_matrix_min12_max25.csv", header=0, index_col=0)
    # print("\nAdjacency matrix:\n", adjacency_matrix)

    # n_edges = np.sum(adjacency_matrix.values > 0)
    # print("Number of edges:", n_edges)

    ## 'final_input' is the file that contains all transcriptomics and metagenomics data
    MG_T_Ph = pd.read_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/T_MG_P_input_data/final_input.csv", header=0, index_col=0)
    # get the IDs of the sample with metagenomics and transcriptomics data
    samples_with_MG_T_Ph_data = list(MG_T_Ph.index)
    #print(samples_with_MG_T_Ph_data)

    ## Study correlations between features in matrix MG_T
    # plot_feature_correlation(MG_T)
    
    MG_T_Ph_filteredSamples = MG_T_Ph.loc[adjacency_matrix.index] # get the samples with metagenomics and transcriptomics data that are in the adjacency matrix

    Pheno = MG_T_Ph_filteredSamples[['weight']]
    # print("\nPhenotype:\n", list(Pheno.columns))
    y = torch.tensor(Pheno.values, dtype=torch.float)

    MG_T = MG_T_Ph_filteredSamples.loc[:, ~MG_T_Ph_filteredSamples.columns.isin(['size', 'weight'])]
    print(MG_T)

    x = torch.tensor(MG_T.values, dtype=torch.float)


    ### BUILDING THE GRAPH
    edge_index, edge_attr, sample_to_index_df = get_edges_from_adjacency(adjacency_matrix, print_tmp=True)
    
    ## Save to csv
    edge_index_df = pd.DataFrame(edge_index.numpy().T, columns=['source', 'target'])
    sample_to_index_df.to_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/output/sample_to_index_2.csv", index=False)
    edge_index_df.to_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/output/edge_index_2.csv", index=True)


    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=MG_T.shape[0], num_node_features=MG_T.shape[1])

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    # print(f'Number of training nodes: {data.train_mask.sum()}')
    # print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')



    # Save it to the correct path. Next time you can skip the process(). 
    # torch.save(self.collate([data]), self.processed_paths[0])
    # print(f"Saved data to {self.processed_paths[0]}")


    print("\nedge_index\n", edge_index)
    print(edge_index.shape)

    print("\nedge_attr\n", edge_attr)
    print(edge_attr.shape)
    print("\nGRAPH:\n", data)

    print("\nNumber of nodes in the graph:", data.num_nodes)



    # dataset_try = TUDataset(root='./data/', name='MUTAG')
    # data_try = dataset_try[0]


    class GCN(torch.nn.Module):
        def __init__(self, num_node_features, state_dim):
            super(GCN, self).__init__()
            self.num_node_features = num_node_features
            self.state_dim = state_dim
            
            self.linear1 = torch.nn.Linear(self.num_node_features, self.state_dim)
            self.conv1 = GCNConv(self.state_dim, self.state_dim)
            self.conv2 = GCNConv(self.state_dim, self.state_dim)
            self.linear2 = torch.nn.Linear(self.state_dim,1)
        
        
        def forward(self, x, edge_index):
            # print("1: ", x.shape)
            x = self.linear1(x)
            # print("2: ", x.shape)
            x = self.conv1(x, edge_index)
            # print("3: ", x.shape)
            x = F.relu(x)
            # print("4: ", x.shape)
            # x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            # print("5: ", x.shape)
            x = self.linear2(x)
            # print("6: ", x.shape)

            # print("\n\nX: \n", x)

            return x #F.log_softmax(x, dim=1)
        
    device = 'cpu'
    model = GCN(num_node_features=data.num_node_features, state_dim=64)#.to(device)
    print(model)


    data = data#.to(device)

    # Loss function
    # mse_loss = F.mse_loss()
    mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)#, weight_decay=5e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)


    def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = mse_loss(out, data.y)  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      scheduler.step() # Update the learning rate.
      return loss

    epochs = 100000


    for epoch in range(epochs):
        loss = train()
        if epoch % 1000 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')



    
    # for epoch in range(epochs):

    #     model.train()
        
    #     out = model(data.x)
    #     # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    #     # loss = F.mse_loss(out.squeeze(), data.y.squeeze())
    #     loss = mse_loss(out.squeeze(), data.y.squeeze())

    #     # Gradient step
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     if epoch % 1000 == 0:
    #         print(f'Epoch: {epoch}, Loss: {loss}')
    #         print(f'- Learning rate   = {scheduler.get_last_lr()[0]:.1e}')



    #     # Learning rate scheduler step
    #     scheduler.step()


    #     # HERE ADD VALIDATION LOSS COMPUTATION AND THEN PRINT

    breakpoint()

    model.eval()
    out = model(data)

    print(data.y)






if __name__ == "__main__":
    main()