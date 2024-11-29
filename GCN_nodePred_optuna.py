import optuna

from torch_geometric.data import Data
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, TUDataset
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import os
from sklearn.model_selection import train_test_split

from similarity_graph_utilities import get_edges_from_adjacency, plot_feature_correlation
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

import random

from torch.nn.functional import elu
import csv
import pandas as pd

"""
 Implementing a Graph Convolutional Network (GCN) for node classification
 THIS VERSION USES OPTUNA FOR HYPERPARAMETER TUNING
 it was copied from GCN_nodePred_similarityMatrix.py on 26.11.2024.

 INPUTS:
    - adjacency_matrix: matrix n_samples x n_samples whose values represent similarities.
    - MG_T: matrix n_samples x n_features whose values represent METAGENOMIC and TRANSCRIPTOMICS FEATURES of the samples.
    - Pheno: matrix n_samples x 1 whose values represent the PHENOTYPE of the samples.
    
"""

# Regularization functions that account for connectivity: Use 5 or 10 as minimum nunmber of neighbors

# Things to try: Loss functions (MAE; MSE); Optimizers; Layers


# WEIGHTED adjacency matrix:
ADJ_MATRIX_FILE = "data/adj_matrices/adj_matrix_hc_40PCs_rv_0.4.csv"
# NON-weighted adjacency matrix:

MG_T_FILE = "data/T_MG_final_scaled.csv"

# Decide whether you want to restrict the feature space to T or MG only
T_ONLY = False
MG_ONLY = False

T_SELECTION_ALGORITHM = "Variance" # Lasso, Variance, Autoencoder, PCA, scaled_Variance
    


    # parser = argparse.ArgumentParser(description="Run parts of the script")
    # parser.add_argument("--mode", choices=["optuna", "evaluate"], required=True, help="Select the mode to run")  
    # args = parser.parse_args()

    # if args.mode == "optuna":
    #     run_optuna()
    # elif args.mode == "evaluate":
    #     best_params = {  # Replace with actual best parameters
    #         'param1': value1,
    #         'param2': value2,
    #     }
    #     evaluate_model(best_params)




def main():





    
    adjacency_matrix = pd.read_csv(ADJ_MATRIX_FILE, header=0, index_col=0)
    file_name = os.path.basename(ADJ_MATRIX_FILE)

    samples_ids = adjacency_matrix.index
    
    print("\nUsing adjacency matrix: ", file_name)
    # print("Number of edges:", np.sum(adjacency_matrix.values > 0))



    ### INPUT: METAGENOMIC AND TRANSCRIPTOMIC FEATURES -> X
    
    ### T10 (10 features from Dylan and)
    # MG_T_Ph_f = pd.read_csv("data/T_MG_P_input_data/final_input.csv", header=0, index_col=0)
    # MG_T_Ph_filteredSamples = MG_T_Ph_f.loc[adjacency_matrix.index] # get the samples with metagenomics and transcriptomics data that are in the adjacency matrix

    # MG_T = pd.read_csv("data/T_MG_P_input_data/scaled_input.csv", header=0, index_col=0)
    # MG_T_filteredSamples = MG_T_Ph.loc[adjacency_matrix.index]
    # MG_T = MG_T_filteredSamples
    # MG = MG_T_Ph_f.loc[:, MG_T_Ph_filteredSamples.columns.str.startswith('MAG')]
    
    # ### METAGENOME
    # MG = pd.read_csv("data/HoloFish_MetaG_MeanCoverage_20221114.csv", header=0, index_col=0)
    # MG = MG.loc[samples_ids, MG.columns.str.startswith('MAG')] 


    # def normalize_data(data):
    #     sample_ids = data.index
    #     feature_names = data.columns
    #     data_matrix = data.values
    #     scaler = MinMaxScaler()
    #     scaled_matrix = scaler.fit_transform(data_matrix)
    #     scaled_data = pd.DataFrame(scaled_matrix, index=sample_ids, columns=feature_names)
    #     return scaled_data

    # MG = normalize_data(MG)
    
    # # OLD NORMALIZATION:
    # # MG = pd.DataFrame(normalize(MG, norm='l2', axis=1), index=MG.index, columns=MG.columns)
    # # standard_scaler = StandardScaler()
    # # MG = pd.DataFrame(standard_scaler.fit_transform(MG), index=MG.index, columns=MG.columns)


    # ### TRANSCRIPTOME
    # ## 10 features from Dylan
    # # T_10 = MG_T_Ph_filteredSamples.loc[:,~MG_T_Ph_filteredSamples.columns.str.startswith(('MAG', 'weight', 'size'))] # Keeping the 10 original (from Dylan) transcriptomic features
    # ## T features selected by Lasso/Variance/Autoencoder/PCA
    
    # T_path = "T_selected_features_" + T_SELECTION_ALGORITHM + ".csv"
    # T_selected = pd.read_csv(os.path.join("data/T_features/", T_path), header=0, index_col=0) 

    # # Remove the T features that are in both T_10 and T_selected
    # # T_selected = T_selected.rename(columns=lambda x: x.split("+")[-1], inplace=False)
    # # columns_toremove = T_selected.columns.intersection(T_10.columns)
    # # T_selected_filtered = T_selected.drop(columns=columns_toremove)

    # # T = pd.concat([T_10, T_selected_filtered], axis=1)
    # T = T_selected
    
    # T = normalize_data(T)

    # breakpoint()


    # # OLD NORMALIZATION:
    # # T = pd.DataFrame(normalize(T, norm='l2', axis=1), index=T.index, columns=T.columns)
    # # standard_scaler = StandardScaler()
    # # T = pd.DataFrame(standard_scaler.fit_transform(T), index=T.index, columns=T.columns)

    # # Getting Metagenomics and Transcriptomics features in one single object
    # MG_T = MG if MG_ONLY else T if T_ONLY else pd.concat([MG, T], axis=1)
    
    # # MG_T.to_csv("data/output/MG_T_normalized.csv", index=True)

    # # MG_T = MG_T.dropna()
    # # MG_T.to_csv("data/output/MG_T_normalized_removeNa.csv", index=True)

    # # MG_T = MG_T.loc[samples_ids]
    # # MG_T.to_csv("data/output/MG_T_normalized_SampleIDcheck.csv", index=True)
    # # # breakpoint()

    # Read the saved file

    MG_T = pd.read_csv(MG_T_FILE, header=0, index_col=0, sep=',')

    ### Study correlations between features in matrix MG_T
    # plot_feature_correlation(T)


    X = torch.tensor(MG_T.values, dtype=torch.float32)
    # X = torch.tensor(MG_T_normalized, dtype=torch.float)
    print("X shape (features selected from our script): ", X.shape)


    ##### IMPORTANT: USING KEGG PATHWAYS AS FEATURES
    # T = pd.read_csv("data/kegg/fatty_scaled.csv", header=0, index_col=0)
    # T = T.loc[samples_ids]

    # T_MG = pd.read_csv("data/kegg/scaled_input_carbo.csv", header=0, index_col=0) # scaled_input_fatty.csv, scaled_input_carbo.csv, scaled_input_lipid.csv
    # T_MG = T_MG.loc[samples_ids]
    # X = torch.tensor(T_MG.values, dtype=torch.float32)
    # print("X shape (features selected from KEGG): ", X.shape)




    ### PHENOTYPE -> y

    Pheno = pd.read_csv("data/HoloFish_FishVariables_20221116.csv", header=0, index_col=0)
    Pheno = Pheno.loc[samples_ids, "Gutted.Weight.kg"]


    # Pheno = MG_T_Ph_filteredSamples[['weight']] 
    y = torch.tensor(Pheno.values, dtype=torch.float32)


    ### BUILDING THE GRAPH
    edge_index, edge_attr, sample_to_index_df = get_edges_from_adjacency(adjacency_matrix, print_tmp=False)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)


    # edge_index_df = pd.DataFrame(edge_index.numpy().T, columns=['source', 'target'])
    # sample_to_index_df.to_csv("data/output/sample_to_index_2.csv", index=False)
    # edge_index_df.to_csv("data/output/edge_index_2.csv", index=True)

    num_nodes=MG_T.shape[0]
    num_node_features=MG_T.shape[1]

    ### Split into training and validation
    # rng = torch.Generator().manual_seed(0)
    # train_dataset, validation_dataset = random_split(data, (132, 57), generator=rng) # 189 nodes
    # # train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

    # train_loader = DataLoader(train_dataset, batch_size=132)
    # # validation_loader = DataLoader(validation_dataset, batch_size=57)
    # # test_loader = DataLoader(test_dataset, batch_size=44)


    # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)    
    # X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(X, y, np.arange(num_nodes), test_size=0.20, random_state=42)
    # print("X_train shape: ", X_train.shape)
    # print("X_val shape: ", X_val.shape)
    # # Create masks for training and validation sets
    # train_mask = np.zeros(num_nodes, dtype=bool)
    # val_mask = np.zeros(num_nodes, dtype=bool)
    # train_mask[train_idx] = True
    # val_mask[val_idx] = True


    ### Split into training and validation - 5-fold CROSS-VALIDATION
    # Shuffle the list of numbers
    random.seed(2)
    shuffled_list = list(range(0, num_nodes))
    random.shuffle(shuffled_list) 


    NUMBER_RANDOM_SPLITS = 5 # No. CROSS-VALIDATION (CV) splits
    all_lists = {}
    for i in range(NUMBER_RANDOM_SPLITS):
        list_name = str("list" + str(i))
        all_lists[list_name] = []

    for i in range(0, num_nodes):
        all_lists[str("list" + str(i%NUMBER_RANDOM_SPLITS))] = all_lists[str("list" + str(i%NUMBER_RANDOM_SPLITS))] + [shuffled_list[i]] # assing iteratively the elements of the shuffled list to the different lists
  
    # alternating using one of the lists (ith_list) as validation and the rest (all_lists_except_ith) as training
    # for i in range(0, NUMBER_RANDOM_SPLITS):
    i=0

    # print("\n\n\n\n Cross-Validation: ", str(i+1), "/",NUMBER_RANDOM_SPLITS ,"\n\n")
    ith_list = all_lists["list" + str(i)]
    all_lists_except_ith = [all_lists[key] for key in all_lists if key != "list" + str(i)]
    union_lists = [element for sublist in all_lists_except_ith for element in sublist]

    train_mask = np.zeros(num_nodes, dtype=bool) #it was 'bool' instead of 'torch.bool'
    val_mask = np.zeros(num_nodes, dtype=bool) #bool
    train_mask[union_lists] = True
    val_mask[ith_list] = True

    print("Number of training samples: ", len(union_lists))
    print("Number of validation samples: ", len(ith_list))


    ### GRAPH OBJECT
    data = Data(x=X, edge_index=edge_index, edge_weight=edge_attr, y=y, num_nodes=num_nodes, num_node_features=num_node_features)
    

    ## Statistics
    print(f'\nNumber of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    # print(f'Number of training nodes: {data.train_mask.sum()}')
    # print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}\n')


    # Save it to the correct path. Next time you can skip the process(). 
    # torch.save(self.collate([data]), self.processed_paths[0])
    # print(f"Saved data to {self.processed_paths[0]}")

    # print("\nGRAPH:\n", data)
    # print("\nedge_index\n", edge_index)

    # dataset_try = TUDataset(root='./data/', name='MUTAG')
    # data_try = dataset_try[0]


    class GCN_weighted(torch.nn.Module):
        def __init__(self, num_node_features, hidden_dim1, hidden_dim2):

            super(GCN_weighted, self).__init__()
            torch.manual_seed(12)

            self.num_node_features = num_node_features
            self.hidden_dim1 = hidden_dim1
            self.hidden_dim2 = hidden_dim2

            self.conv1 = GCNConv(self.num_node_features, self.hidden_dim1)
            self.conv2 = GCNConv(self.hidden_dim1, self.hidden_dim2)
            self.linear = torch.nn.Linear(self.hidden_dim2,1)

        def forward(self, x, edge_index, edge_weight):
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
            
            x = F.elu(x)
            # x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
            
            x = F.elu(x) # TODO CHECK IF WE NEED TO REMOVE THIS
            x = self.linear(x)
            
            return x

    class GCN(torch.nn.Module):
        def __init__(self, num_node_features, state_dim):
            super().__init__()
            # super(GCN, self).__init__()
            torch.manual_seed(12) #random seed

            self.num_node_features = num_node_features
            self.state_dim = state_dim

            self.conv1 = GCNConv(self.num_node_features, self.state_dim)
            self.conv2 = GCNConv(self.state_dim, int(self.state_dim/2))
            self.linear = torch.nn.Linear(int(self.state_dim/2),1)

        def forward(self, x, edge_index, edge_weight):
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            # x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            x = self.linear(x)

            return x
        
    class GCN_2(torch.nn.Module):
        def __init__(self, num_node_features, state_dim):
            super().__init__()
            torch.manual_seed(12) #random seed

            self.num_node_features = num_node_features
            self.state_dim = state_dim

            self.conv1 = GCNConv(self.num_node_features, self.state_dim)
            # self.conv2 = GCNConv(self.state_dim, self.state_dim)
            self.conv2 = GCNConv(self.state_dim, int(self.state_dim/2))
            self.conv3 = GCNConv(int(self.state_dim/2), int(self.state_dim/2))
            self.linear = torch.nn.Linear(int(self.state_dim/2),1)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = self.conv2(x, edge_index)
            x = F.elu(x)
            x = self.conv3(x, edge_index)
            # x = x.relu()
            # x = self.conv4(x, edge_index)
            x = self.linear(x)


            return x

    class GCN_deep(torch.nn.Module):
        def __init__(self, num_node_features, state_dim):
            super(GCN_deep, self).__init__()
            torch.manual_seed(12) #random seed

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
            # self.linear5 = torch.nn.Linear(self.state_dim,self.state_dim)

            # self.conv6 = GCNConv(self.state_dim, self.state_dim)
            # self.linear6 = torch.nn.Linear(self.state_dim,self.state_dim)
            
            # self.conv7 = GCNConv(self.state_dim, self.state_dim)
            # self.linear7 = torch.nn.Linear(self.state_dim,self.state_dim)

            # self.conv8 = GCNConv(self.state_dim, self.state_dim)

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
            # x = self.linear4(x)
            # x = self.conv5(x, edge_index)
            # x = F.relu(x)
            # x = F.dropout(x, p=0.2, training=self.training)
            # x = self.linear5(x)

            # x = self.conv6(x, edge_index)
            # x = F.relu(x)
            # x = F.dropout(x, p=0.5, training=self.training)
            # x = self.linear6(x)
            # x = self.conv7(x, edge_index)
            # x = F.relu(x)
            # x = F.dropout(x, p=0.5, training=self.training)
            # x = self.linear7(x)

            # x = self.conv8(x, edge_index)
            # x = F.relu(x)
            
            x = self.linear_final(x)

            # print("\n\nX: \n", x)

            return x #F.log_softmax(x, dim=1)
        

    def objective(trial):
        hidden_dim1 = trial.suggest_int('hidden_dim1', 16, 64)
        hidden_dim2 = trial.suggest_int('hidden_dim2', 16, 48)
        lr = trial.suggest_float('lr', 2e-6, 5e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])

        device = 'cpu'
        model = GCN_weighted(num_node_features=data.num_node_features, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2).to(device)
        # model = GCN(hidden_channels=64).to(device)
        print(model)

        # data = Data(x=data.x, y=data.y)

        # data = data.to(device) GIVES ERROR

        loss_fun = torch.nn.MSELoss()
        # loss_fun = torch.nn.L1Loss()

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)            
        
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999) # !!! WE REMOVE THIS BECAUSE LR is ALREADY OPTIMIZED WITH OPTUNA

        patience = 10
        best_val_loss = float('inf')
        epochs_without_improvement = 0


        def train(losses_train, losses_val):
            model.train() # Set the model to training mode.
            optimizer.zero_grad()  # Clear gradients.
            out = model(data.x, data.edge_index, data.edge_weight)  # Perform a single forward pass. ### TODO CHECK, here we use the entire dataset, not just the training set. 
            out = out.squeeze()

            loss_train = loss_fun(out[train_mask], data.y[train_mask])
            losses_train.append(loss_train.item())
            
            loss_val = loss_fun(out[val_mask], data.y[val_mask])
            losses_val.append(loss_val.item())

            loss = loss_fun(out, data.y)
        #   loss = loss_fun(out.squeeze(), data.y.squeeze())


            # TODO IN THE FOLLOWING LINES: ONLY DO optimizer.step() IF VAL_LOSS IS DECREASING. SHOULD WE ALSO SHUFFLE TRAINING AND VALIDATION SETS IN THIS PROCESS?
            ## Only compute gradients for nodes in the training set
        #   if len(losses_val) > 1 and losses_val[-1] <= losses_val[-2]:

            
            loss_train.backward()  # Derive gradients.  ### TODO CHECK, here we only use the training set.
            optimizer.step()  # Update parameters based on gradients.
            # scheduler.step() # Update the learning rate.

            return loss, losses_train, losses_val




        # variance = torch.var(data.y)
        # print("Variance of y:", variance)
        epochs = 3000

        # print(y)
        # out = model(data)
        # print(out)

        losses_train, losses_val = [], []

        for epoch in range(epochs):

            loss, losses_train, losses_val = train(losses_train, losses_val)           
            val_loss = losses_val[-1]
            # breakpoint()
            
            trial.report(val_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = model.state_dict()
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                raise optuna.TrialPruned()       


            # if epoch % 100 == 0:
            #     print(f'Epoch: {epoch:03d}, Learning rate: {scheduler.get_last_lr()[0]:.1e}, Train loss: {losses_train[-1]:.4f}, Validation loss: {losses_val[-1]:.4f}')


        # ### EVALUATION
        # model.eval()
        # out = model(data.x, data.edge_index, data.edge_weight)
        # out = out.squeeze()


        # def save_values(out, y, file_name, i, mask=None):
        #     if val_mask is not None:
        #         out_array_df = pd.DataFrame(out[val_mask].detach().numpy())
        #         out_array_df.to_csv("data/predictions/pred_" + file_name + "_CV{}.csv".format(i+1), index=False, header=False)
        #         y_df= pd.DataFrame(y[val_mask].detach().numpy())
        #         y_df.to_csv("data/y_values/true_" + file_name + "_CV{}.csv".format(i+1), index=False, header=False)
        #     else:
        #         out_array_df = pd.DataFrame(out.detach().numpy())
        #         out_array_df.to_csv("data/predictions/pred_" + file_name + "_CV.csv", index=False, header=False)
        #         y_df= pd.DataFrame(y.detach().numpy())
        #         y.to_csv("data/y_values/true_" + file_name + "_CV.csv", index=False, header=False)
            
        
        # save_values(out, data.y, file_name, i, mask=val_mask)

        
        # # for predicted, actual  in zip(out, data.y): print(f'Predicted: {predicted.item():.4f}, Actual: {actual.item():.4f}, Difference: {abs(predicted.item()-actual.item()):.4f}')
        # print("Avg difference: \n",np.mean(np.abs(out.detach().numpy() - data.y.detach().numpy())))
        # print('Max difference: \n', np.max(np.abs(out.detach().numpy() - data.y.detach().numpy())))

        # # print("Avg difference (train): \n",np.mean(np.abs(out[train_mask].detach().numpy() - data.y[train_mask].detach().numpy())))
        # print("Avg difference (val): \n",np.mean(np.abs(out[val_mask].detach().numpy() - data.y[val_mask].detach().numpy())))
        # print('Max difference (val): \n', np.max(np.abs(out[val_mask].detach().numpy() - data.y[val_mask].detach().numpy())))


        # ### Plotting the loss over epochs
        # plt.plot(range(epochs), losses_train, label='Training loss')
        # plt.plot(range(epochs), losses_val, label='Validation loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title('Training and Validation Loss')
        # plt.ylim(0, 4)  # Set the y-axis limits
        # plt.legend()
        # # plt.show()
        # # breakpoint()


        # name_fig_base = "data/losses/" + str(file_name)[:-4]

        # if MG_ONLY: name_fig = name_fig_base + "_" "MG_only" +  "_loss_CV{}.png".format(i+1)
        # else:
        #     name_fig = name_fig_base + "_" + str(T_SELECTION_ALGORITHM) + str(MG_T.shape[1])
        #     if T_ONLY: name_fig = name_fig + "_T_only"
        #     name_fig = name_fig + "_loss_CV{}.png".format(i+1)

        # # 
        # # name_fig = "data/losses/" + str(file_name)[:-4] + "_" + str(T_SELECTION_ALGORITHM) + str(T.shape[1]) + "_T_only"  "_loss_CV{}.png".format(i+1)


        # plt.savefig(str(name_fig))

        # plt.close()

        return best_val_loss
        # until here, INSIDE THE TRIAL

        


    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    
    # study.optimize(objective, n_trials=100)

    best_params = {'hidden_dim1': 55, 'hidden_dim2': 30, 'lr': 0.0005424950447117308, 'weight_decay': 0.0005325208317546223, 'optimizer': 'adamw'}

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_params = trial.params
    final_model = GCN_weighted(num_node_features=data.num_node_features, hidden_dim1=best_params['hidden_dim1'], hidden_dim2=best_params['hidden_dim2'])
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    final_model.train()

    best_model_state = None
    best_loss_val = float('inf')

    losses_train = []
    losses_val = []

    for epoch in range(5000):

        # for data in train_loader:
        final_optimizer.zero_grad()
        
        out = final_model(data.x, data.edge_index, data.edge_weight)

        out = out.squeeze()

        loss_fun = torch.nn.MSELoss()

        loss_train = loss_fun(out[train_mask], data.y[train_mask])
        losses_train.append(loss_train.item())
        
        loss_val = loss_fun(out[val_mask], data.y[val_mask])
        losses_val.append(loss_val.item())


        loss_train.backward()
        final_optimizer.step()

        final_model.eval()
        with torch.no_grad():
            out = final_model(data.x, data.edge_index, data.edge_weight)
            out = out.squeeze()

            out_array_df = pd.DataFrame(out[val_mask].detach().numpy())
            # out_array_df.to_csv("data/predictions/pred_" + file_name + "_CV{}.csv".format(i+1), index=False, header=False)
            y_df= pd.DataFrame(y[val_mask].detach().numpy())
            # y_df.to_csv("data/y_values/true_" + file_name + "_CV{}.csv".format(i+1), index=False, header=False)

        if epoch % 100 == 0:
            print(f'Epoch: {epoch:03d}, Train loss: {losses_train[-1]:.4f}, Validation loss: {losses_val[-1]:.4f}')

        if loss_val < best_loss_val:
            best_loss_val = loss_val
            best_model_state = final_model.state_dict()

    final_model.load_state_dict(best_model_state)

    breakpoint()
    # !! CV for cycle ends here







if __name__ == "__main__":
    main()





    
