"""
Select features from the transcriptome data using different feature selection methods.
For now, we select the Transcriptome features independently from the Metagenome features.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from scipy.spatial import distance
# from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
import torch.utils.data as Data


SELECTION_METHOD =  "GS" #GS "variance" # "kegg"#"MOGCN-VAE" # "Variance", "Autoencoder", "PCA", "Lasso", "MOGCN-VAE"

# GS: METHOD like VARIANCE but that reduced MULTICOLINEARITY (e.g. PCA, but with real features, not PCs)
# with LASSO, the T features are selected to predict Pheno for the overall dataset, this is overfitting, because we want that they are equally good to predict Pheno for any of the samples, not just those in the main cluster

# TODO For clusters (sometimes with size 1) where all samples are missing that feature, add the average of nearby clusters/samples

N_FEATURES_TO_USE = 100 # VARIANCE and GS methods only
ALPHA = 0.03            # LASSO method only


def normalize_data(data):
    sample_ids = data.index
    feature_names = data.columns
    data_matrix = data.values
    scaler = MinMaxScaler()
    scaled_matrix = scaler.fit_transform(data_matrix)
    scaled_data = pd.DataFrame(scaled_matrix, index=sample_ids, columns=feature_names)
    return scaled_data

def fill_NaN(data):
    # Fill NaN values with the average of the same column of the other non NaN elements of the cluster # for each NaN value in G_T_MG, input the average for the same column of the other non NaN elements of the cluster (same "cluster_id")
    for cluster_id in data['cluster_id'].unique():
        cluster_data = data[data['cluster_id'] == cluster_id]
        for feature in data.columns:
            if feature != 'cluster_id':
                mean_value = cluster_data[feature].mean()
                data.loc[data['cluster_id'] == cluster_id, feature] = data.loc[data['cluster_id'] == cluster_id, feature].fillna(mean_value)
    return data

def combine_G_T_MG(sample_pcs, T, MG):

    G_T_MG = pd.concat([sample_pcs, T, MG], axis=1)
    G_T_MG = G_T_MG.sort_values(by='cluster_id')
    G_T_MG.to_csv("data/debug/G_T_MG.csv", index=True, sep=',') # DEBUG, this csv will contain NaN values for the samples that do not have T or MG data

    G_T_MG = fill_NaN(G_T_MG) 
    G_T_MG.to_csv("data/debug/G_T_MG_filled.csv", index=True, sep=',') # DEBUG


    G_T_MG = G_T_MG.dropna() # this can be moved inside the function fill_NaN, but we keep it here for debugging purposes
    # number_of_removed_samples = G_T_MG.shape[0] - sample_pcs.shape[0]
    G_T_MG.to_csv("data/debug/G_T_MG_filled_no_NaN.csv", index=True, sep=',') # DEBUG

    return G_T_MG

def extract_and_save_G(data):
    G = data.loc[:, data.columns.str.startswith('PC') | data.columns.str.startswith('cluster_id')]
    G.to_csv("data/G_processed.csv", index=True, sep=',')
    print("saving OUTPUT with G for selected samples: data/G_processed.csv")
    return G

def extract_and_save_T_MG(data):
    T_MG = data.loc[:, ~data.columns.str.startswith('PC') & ~data.columns.str.startswith('cluster_id')]
    T_MG.to_csv("data/T_MG_processed.csv", index=True, sep=',')
    print("saving OUTPUT with selected features T_MG: data/T_MG_processed.csv")
    return T_MG


# TRY Multi-Modal autoencoder from MoGCN paper:

class MMAE(nn.Module):
    def __init__(self, in_feas_dim, latent_dim, a=0.4, b=0.3, c=0.3):
        '''
        :param in_feas_dim: a list, input dims of omics data
        :param latent_dim: dim of latent layer
        :param a: weight of omics data type 1
        :param b: weight of omics data type 2
        # :param c: weight of omics data type 3
        '''
        super(MMAE, self).__init__()
        self.a = a
        self.b = b
        # self.c = c
        self.in_feas = in_feas_dim
        self.latent = latent_dim

        #encoders, multi channel input
        self.encoder_omics_1 = nn.Sequential(
            nn.Linear(self.in_feas[0], self.latent),
            nn.BatchNorm1d(self.latent),
            nn.Sigmoid()
        )
        self.encoder_omics_2 = nn.Sequential(
            nn.Linear(self.in_feas[1], self.latent),
            nn.BatchNorm1d(self.latent),
            nn.Sigmoid()
        )
        # self.encoder_omics_3 = nn.Sequential(
        #     nn.Linear(self.in_feas[2], self.latent),
        #     nn.BatchNorm1d(self.latent),
        #     nn.Sigmoid()
        # )
        #decoders
        self.decoder_omics_1 = nn.Sequential(nn.Linear(self.latent, self.in_feas[0]))
        self.decoder_omics_2 = nn.Sequential(nn.Linear(self.latent, self.in_feas[1]))
        # self.decoder_omics_3 = nn.Sequential(nn.Linear(self.latent, self.in_feas[2]))

        #Variable initialization
        for name, param in MMAE.named_parameters(self):
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0, std=0.1)
            if 'bias' in name:
                torch.nn.init.constant_(param, val=0)

    def forward(self, omics_1, omics_2):#, omics_3):
        '''
        :param omics_1: omics data 1
        :param omics_2: omics data 2
        :param omics_3: omics data 3
        '''
        encoded_omics_1 = self.encoder_omics_1(omics_1)
        encoded_omics_2 = self.encoder_omics_2(omics_2)
        # encoded_omics_3 = self.encoder_omics_3(omics_3)
        latent_data = torch.mul(encoded_omics_1, self.a) + torch.mul(encoded_omics_2, self.b)# + torch.mul(encoded_omics_3, self.c)
        decoded_omics_1 = self.decoder_omics_1(latent_data)
        decoded_omics_2 = self.decoder_omics_2(latent_data)
        # decoded_omics_3 = self.decoder_omics_3(latent_data)
        return latent_data, decoded_omics_1, decoded_omics_2#, decoded_omics_3

    def train_MMAE(self, train_loader, learning_rate=0.001, device=torch.device('cpu'), epochs=100):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        loss_ls = []


        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9970)
        
        
        for epoch in range(epochs):
            train_loss_sum = 0.0       #Record the loss of each epoch
            for (x,y) in train_loader:

                omics_1 = x[:, :self.in_feas[0]]
                omics_2 = x[:, self.in_feas[0]:self.in_feas[0]+self.in_feas[1]]

                # omics_3 = x[:, self.in_feas[0]+self.in_feas[1]:self.in_feas[0]+self.in_feas[1]+self.in_feas[2]]

                omics_1 = omics_1.to(device)
                omics_2 = omics_2.to(device)
                # omics_3 = omics_3.to(device)


                # latent_data, decoded_omics_1, decoded_omics_2, decoded_omics_3 = self.forward(omics_1, omics_2, omics_3)
                latent_data, decoded_omics_1, decoded_omics_2 = self.forward(omics_1, omics_2)
                # loss = self.a*loss_fn(decoded_omics_1, omics_1)+ self.b*loss_fn(decoded_omics_2, omics_2) + self.c*loss_fn(decoded_omics_3, omics_3)
                loss = self.a*loss_fn(decoded_omics_1, omics_1)+ self.b*loss_fn(decoded_omics_2, omics_2) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()

                train_loss_sum += loss.sum().item()
            
            scheduler.step()

            loss_ls.append(train_loss_sum)
            print(f'epoch: {epoch + 1} | lr: {scheduler.get_last_lr()[0]:.1e} | loss: {train_loss_sum} ')   

            #save the model every 10 epochs, used for feature extraction
            if (epoch+1) % 10 ==0:
                torch.save(self, 'model/model_{}.pkl'.format(epoch+1))
                # torch.save(self, 'model/AE/model_{}.pkl'.format(epoch+1))

        #draw the training loss curve
        plt.plot([i + 1 for i in range(epochs)], loss_ls)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('AE_train_loss.png')
        # plt.savefig('result/AE_train_loss.png')


def main():

    ### get the IDs of the sample with genomics data
    # sample_pcs = pd.read_excel("data/PCA/principalComponents_ofFish_basedOnGWAS.xlsx", header=0, index_col=0)    
    sample_pcs = pd.read_csv("data/PCA/PCs_Fish_GWAS-based_cluster-filtered.csv", header=0, index_col=0)
    sample_pcs_index = list(sample_pcs.index)
    ### get the IDs of the sample with metagenomics and transcriptomics data
    # samples_with_MG_T_Ph_data = list((pd.read_csv("data/T_MG_P_input_data/final_input.csv", header=0, index_col=0)).index)
    # sample_index_final = sample_pcs_index[sample_pcs_index.isin(samples_with_MG_T_Ph_data)]
    # We removed the above lines: NOW WE SEPARATELY PROCESS MG and T! (see also clustering.py)


    ### METAGENOME features
    MG = pd.read_csv("data/HoloFish_MetaG_MeanCoverage_20221114.csv", header=0, index_col=0)
    MG = MG.loc[:, MG.columns.str.startswith('MAG')]
    MG = MG.loc[MG.index.isin(sample_pcs_index)] # get the samples for which we have G

    ### TRANSCRIPTOME features (UNFILTERED)
    T_unfiltered = pd.read_csv("data/HoloFish_HostRNA_normalised_GeneCounts_230117.csv", header=0, index_col=0)
    T_unfiltered = T_unfiltered[T_unfiltered.index.isin(sample_pcs_index)]

    # MG = MG.loc[MG.index.isin(T_unfiltered.index)] # REMOVED, NOW WE FILL MISSING VALUES WITH THE AVERAGE OF THE CLUSTER (see clustering.py) 
    
    # PHENOTYPE (only used for the LASSO method)
    Pheno = pd.read_csv("data/HoloFish_FishVariables_20221116.csv", header=0, index_col=0)
    Pheno = Pheno.loc[T_unfiltered.index, "Gutted.Weight.kg"]
    y = Pheno


    # # Calculate the average of each feature
    # T_feature_avg = T_unfiltered.mean(axis=0)
    # # print(T_feature_avg)


    if SELECTION_METHOD == "variance":
        # Scale and normalize the features
        scaler = StandardScaler()
        T_scaled = scaler.fit_transform(T_unfiltered)

        # Convert the scaled features back to a DataFrame
        T_scaled_df = pd.DataFrame(T_scaled, columns=T_unfiltered.columns, index=T_unfiltered.index)

        # Save the scaled features to a CSV file
        T_scaled_df.to_csv("data/T_features/T_features_scaled.csv", index=True, sep=',')

        
        def select_top_variance_features(data, n_features):
            return data[data.var(axis=0).nlargest(n_features).index]
        
        def save_csv(data, name_modif = ""):
            file_name = "data/T_features/T_selected_features_" + SELECTION_METHOD + name_modif + ".csv"
            data.to_csv(file_name, index=True, sep=',')
            print("saving OUTPUT with selected features: " + file_name)

        # # Select the top N_FEATURES_TO_USE with the highest variance
        T_var_selected = select_top_variance_features(T_unfiltered, N_FEATURES_TO_USE)
        # save_csv(T_var_selected)
        
        # Select the top N_FEATURES_TO_USE with the highest variance after scaling/normalizing
        T_var_selected_scaled = select_top_variance_features(T_scaled_df, N_FEATURES_TO_USE)
        # save_csv(T_var_selected_scaled, name_modif="_scaled")
        
        T_final = T_var_selected_scaled

    # if SELECTION_METHOD == "explained_variance":
        pass # TODO

    if SELECTION_METHOD == "MOGCN-VAE":

        device=torch.device('cpu')
        bs = 5
        # device = "cpu"

        X = pd.concat([MG, T_unfiltered], axis=1)
        X = X.iloc[:,:].values
        Y = Pheno.values #np.zeros(data.shape[0])
        TX, TY = torch.tensor(X, dtype=torch.float, device=device), torch.tensor(Y, dtype=torch.float, device=device)
        #train a AE model
        Tensor_data = Data.TensorDataset(TX, TY)
        train_loader = Data.DataLoader(Tensor_data, batch_size=bs, shuffle=True)

        in_feas = [MG.shape[1], T_unfiltered.shape[1]]

        mmae = MMAE(in_feas, latent_dim=72, a=0.01, b=0.99, c=0)
        mmae.to(device)
        mmae.train()

        mmae.train_MMAE(train_loader, learning_rate=0.05, device=device, epochs=2000)
        mmae.eval()       #before save and test, fix the variables
        torch.save(mmae, 'model/MMAE_model.pkl')



        # selected_features_df = T_unfiltered[selected_features].sort_index()
        # file_name = "data/T_features/T_selected_features_" + SELECTION_METHOD + ".csv"
        # selected_features_df.to_csv(file_name, index=True, sep=',')

    if SELECTION_METHOD == "PCA":
        pass

    if SELECTION_METHOD == "Autoencoder":
        pass
        # Define the dimensions of the input data
        input_dim = T_unfiltered.shape[1]

        # Define the dimensions of the encoded representation
        ENCODING_DIM = 32

        # Define the input layer
        input_layer = Input(shape=(input_dim,))

        # Define the encoder layers
        encoder = Dense(ENCODING_DIM, activation='relu')(input_layer)

        # Define the decoder layers
        decoder = Dense(input_dim, activation='sigmoid')(encoder)

        # Define the autoencoder model
        autoencoder = Model(inputs=input_layer, outputs=decoder)

        # Compile the autoencoder model
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        # Train the autoencoder model
        autoencoder.fit(T_unfiltered, T_unfiltered, epochs=10, batch_size=32, shuffle=True)

        # Get the encoded features
        encoded_features = encoder.predict(T_unfiltered)

        # Convert the encoded features to a DataFrame
        encoded_features_df = pd.DataFrame(encoded_features, index=T_unfiltered.index)

        # Save the encoded features to a CSV file
        encoded_features_df.to_csv("data/T_features/T_features_encoded.csv", index=True, sep=',')

    if SELECTION_METHOD == "Lasso":

        X = T_unfiltered # Features
        y = Pheno # Target
        
        X = normalize_data(X)
        
        # Create a Lasso regression object with a high alpha value
        lasso = Lasso(alpha=ALPHA, max_iter=1000)

        # Fit the Lasso regression model to the data
        lasso.fit(X, y)

        # Create a SelectFromModel object to select the features with non-zero coefficients
        selection = SelectFromModel(lasso, prefit=True)

        # Get the selected features
        selected_features = X.columns[selection.get_support()]

        print("Number of selected features:", len(selected_features))
        print("\nSelected features:", selected_features)
        
        ## Coefficients of the selected features
        coefficients = lasso.coef_[selection.get_support()]
        print("Coefficients:", coefficients)

        # Save the matrix of the selected features to a CSV file
        selected_features_df = T_unfiltered[selected_features].sort_index()
        file_name = "data/T_features/T_selected_features_" + SELECTION_METHOD + ".csv"
        selected_features_df.to_csv(file_name, index=True, sep=',')

    if SELECTION_METHOD == "kegg": #NOT WORKING
        # Load the KEGG pathways
        # kegg = pd.read_csv("data/T_features/KEGG_pathways.csv", header=0, index_col=0)
        # kegg = kegg.loc[T_unfiltered.columns]
        # kegg = kegg.dropna()

        import requests

        def get_enzymes_for_pathway(pathway_id):
            url = f"https://rest.kegg.jp/link/enzyme/{pathway_id}"
            response = requests.get(url)
            
            # Print the raw response to see if there’s any data returned
            print("Raw response text:")
            print(response.text)  # Show the raw text for debugging


            if response.ok:
                # Process lines and handle potential format issues
                enzymes = []
                for line in response.text.strip().split("\n"):
                    parts = line.split("\t")
                    if len(parts) > 1:
                        enzymes.append(parts[1].strip())  # Extract enzyme ID
                return enzymes
            else:
                print(f"Error fetching enzymes for pathway {pathway_id}")
                return []

        # Example usage for Glycolysis pathway
        pathway_id = "hsa00010"  # KEGG pathway ID for human glycolysis
        enzymes = get_enzymes_for_pathway(pathway_id)

        print("Enzymes in pathway", pathway_id)
        print(enzymes)

    if SELECTION_METHOD == "GS": # Gram-Schmidt orthogonalization

        # Step 0: reduce the number of features by taking the N with highest variance, to reduce computational cost of the GS method
        N_var = 5000 # 19500 -> 5000, then with GS we reduce to N_FEATURES_TO_USE
        def select_top_variance_features(data, n_features):
            return data[data.var(axis=0).nlargest(n_features).index]
        
        T_var_selected = select_top_variance_features(T_unfiltered, N_FEATURES_TO_USE)

        # Step 1: Scale the features
        scaler = StandardScaler()
        T_scaled = scaler.fit_transform(T_unfiltered)
        T_scaled = pd.DataFrame(T_scaled, columns=T_unfiltered.columns, index=T_unfiltered.index)

        # Step 2: Initialize variables
        remaining_features = list(T_scaled.columns)
        selected_features = []
        explained_variance = np.zeros(T_scaled.shape[0])  # Track cumulative explained variance

        # Step 3: Iteratively select features
        for i in range(N_FEATURES_TO_USE):
            max_new_variance = 0
            best_feature = None

            for feature in remaining_features:
                # Compute the orthogonal component of the feature
                residual = T_scaled[feature] - explained_variance
                
                # Compute the variance of the orthogonal component
                new_variance = np.var(residual)

                if new_variance > max_new_variance:
                    max_new_variance = new_variance
                    best_feature = feature

            # Add the best feature to the selected list
            selected_features.append(best_feature)
            # Update explained variance
            explained_variance += T_scaled[best_feature]
            # Remove the feature from the remaining list
            remaining_features.remove(best_feature)

        # Final selected features
        T_final = T_scaled[selected_features]      
        

    # Add MG data to the selected features
    G_T_MG = combine_G_T_MG(sample_pcs, T_final, MG)

    # NOW WE SAVE G AND T_MG SEPARATELY for further processing
    G = extract_and_save_G(G_T_MG)
    T_MG = extract_and_save_T_MG(G_T_MG)           

    # Scale the features (both T and MG) to obtain the final features matrix
    scaler = StandardScaler()
    T_MG_final_scaled = scaler.fit_transform(T_MG)
    T_MG_final_scaled = pd.DataFrame(T_MG_final_scaled, index=T_MG.index, columns=T_MG.columns)
    saving_path = "data/T_MG_final_scaled_"+ SELECTION_METHOD +".csv"
    T_MG_final_scaled.to_csv(saving_path, index=True, sep=',')
    print("saving OUTPUT with selected features T_MG scaled at the end: " + saving_path)

    print("Final T_MG shape:", T_MG_final_scaled.shape)

if __name__ == "__main__":
    main()