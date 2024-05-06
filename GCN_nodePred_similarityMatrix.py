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
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader



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
    
    adjacency_matrix = pd.read_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/adj_matrices/adj_matrix_120PCs_15-20_edges.csv", header=0, index_col=0)
    # print("\nAdjacency matrix:\n", adjacency_matrix)
    
    # n_edges = np.sum(adjacency_matrix.values > 0)
    # print("Number of edges:", n_edges)

    ### PHENOTYPE
    ## 'final_input' is the file that contains Phenotype
    MG_T_Ph_f = pd.read_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/T_MG_P_input_data/final_input.csv", header=0, index_col=0)
    # get the IDs of the sample with phenotype data
    # samples_with_MG_T_Ph_data_f = list(MG_T_Ph_f.index)
    #print(samples_with_MG_T_Ph_data)

    ## Study correlations between features in matrix MG_T
    # plot_feature_correlation(MG_T)
    
    MG_T_Ph_filteredSamples = MG_T_Ph_f.loc[adjacency_matrix.index] # get the samples with metagenomics and transcriptomics data that are in the adjacency matrix
    Pheno = MG_T_Ph_filteredSamples[['weight']]
    # print("\nPhenotype:\n", list(Pheno.columns))
    y = torch.tensor(Pheno.values, dtype=torch.float)

    ### METAGENOMIC AND TRANSCRIPTOMIC FEATURES
    # MG_T = pd.read_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/T_MG_P_input_data/scaled_input.csv", header=0, index_col=0)
    # MG_T_filteredSamples = MG_T_Ph.loc[adjacency_matrix.index]
    # MG_T = MG_T_filteredSamples
    
    MG_T = MG_T_Ph_filteredSamples.loc[:, ~MG_T_Ph_filteredSamples.columns.isin(['size', 'weight'])]
    # print(MG_T)
    x = torch.tensor(MG_T.values, dtype=torch.float)


    ### BUILDING THE GRAPH
    edge_index, edge_attr, sample_to_index_df = get_edges_from_adjacency(adjacency_matrix, print_tmp=False)
    
    ## Save to csv
    edge_index_df = pd.DataFrame(edge_index.numpy().T, columns=['source', 'target'])
    # sample_to_index_df.to_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/output/sample_to_index_2.csv", index=False)
    # edge_index_df.to_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/output/edge_index_2.csv", index=True)


    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=MG_T.shape[0], num_node_features=MG_T.shape[1])

    ## Statistics
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


    # print("\nedge_index\n", edge_index)
    # print(edge_index.shape)

    # print("\nedge_attr\n", edge_attr)
    # print(edge_attr.shape)
    # print("\nGRAPH:\n", data)

    print("\nNumber of nodes in the graph:", data.num_nodes)


    # dataset_try = TUDataset(root='./data/', name='MUTAG')
    # data_try = dataset_try[0]


    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            torch.manual_seed(1234567)
            self.conv1 = GCNConv(data.num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, 1)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    class GCN_deep(torch.nn.Module):
        def __init__(self, num_node_features, state_dim):
            super(GCN_deep, self).__init__()
            torch.manual_seed(1234567)

            self.num_node_features = num_node_features
            self.state_dim = state_dim
            
            self.linear1 = torch.nn.Linear(self.num_node_features, self.state_dim)
            self.conv1 = GCNConv(self.state_dim, self.state_dim)

            self.conv2 = GCNConv(self.state_dim, self.state_dim)
            self.linear2 = torch.nn.Linear(self.state_dim,self.state_dim)

            self.conv3 = GCNConv(self.state_dim, self.state_dim)
            self.linear3 = torch.nn.Linear(self.state_dim,self.state_dim)
            
            self.conv4 = GCNConv(self.state_dim, self.state_dim)
            self.linear4 = torch.nn.Linear(self.state_dim,self.state_dim)
            

            self.conv5 = GCNConv(self.state_dim, self.state_dim)
            self.linear5 = torch.nn.Linear(self.state_dim,self.state_dim)

            self.conv6 = GCNConv(self.state_dim, self.state_dim)
            self.linear6 = torch.nn.Linear(self.state_dim,self.state_dim)
            
            self.conv7 = GCNConv(self.state_dim, self.state_dim)
            self.linear7 = torch.nn.Linear(self.state_dim,self.state_dim)

            self.conv8 = GCNConv(self.state_dim, self.state_dim)

            self.linear_final = torch.nn.Linear(self.state_dim,1)

        def forward(self, x, edge_index):
            # print("1: ", x.shape)
            x = self.linear1(x)
            # print("2: ", x.shape)
            x = self.conv1(x, edge_index)
            # print("3: ", x.shape)
            x = F.relu(x)
            # x = F.dropout(x, p=0.2, training=self.training)
            # print("4: ", x.shape)
            # x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            # print("5: ", x.shape)
            x = F.relu(x)
            # x = F.dropout(x, p=0.2, training=self.training)
            x = self.linear2(x)
            # print("6: ", x.shape)
            x = self.conv3(x, edge_index)
            x = F.relu(x)
            # x = F.dropout(x, p=0.2, training=self.training)
            x = self.linear3(x)
            x = self.conv4(x, edge_index)
            x = F.relu(x)
            # x = F.dropout(x, p=0.2, training=self.training)
            x = self.linear4(x)
            x = self.conv5(x, edge_index)
            x = F.relu(x)
            # x = F.dropout(x, p=0.2, training=self.training)
            x = self.linear5(x)

            x = self.conv6(x, edge_index)
            x = F.relu(x)
            # x = F.dropout(x, p=0.5, training=self.training)
            x = self.linear6(x)
            x = self.conv7(x, edge_index)
            x = F.relu(x)
            # x = F.dropout(x, p=0.5, training=self.training)
            x = self.linear7(x)

            x = self.conv8(x, edge_index)
            x = F.relu(x)
            
            x = self.linear_final(x)

            # print("\n\nX: \n", x)

            return x #F.log_softmax(x, dim=1)
        
    device = 'cpu'
    model = GCN_deep(num_node_features=data.num_node_features, state_dim=24).to(device) #state_dim=16
    # model = GCN(hidden_channels=64).to(device)
    print(model)

    data = data.to(device)

    # Split into training and validation
    # rng = torch.Generator().manual_seed(0)
    # train_dataset, validation_dataset = random_split(data, (180, 27), generator=rng)

    # Create dataloader for training and validation
    # train_loader = DataLoader(train_dataset, batch_size=180)
    # validation_loader = DataLoader(validation_dataset, batch_size=27)
    # test_loader = DataLoader(test_dataset, batch_size=44)


    # Loss function
    # mse_loss = F.mse_loss()
    mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2,weight_decay=1e-3) #weight_decay=5e-4)

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
    
    
    
    # variance = torch.var(data.y)
    # print("Variance of y:", variance)
    epochs = 100000

    # print(y)
    # out = model(data)
    # print(out)

    for epoch in range(epochs):
        loss = train()
        if epoch % 500 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Learning rate: {scheduler.get_last_lr()[0]:.1e}') 

    
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


    model.eval()
    out = model(data.x, edge_index)

    # for predicted, actual  in zip(out, data.y): 
        # print(f'Predicted: {predicted.item():.4f}, Actual: {actual.item():.4f}, Difference: {abs(predicted.item()-actual.item()):.4f}')

    print("Avg difference: \n",np.mean(np.abs(out.detach().numpy() - data.y.detach().numpy())))

    breakpoint()



if __name__ == "__main__":
    main()