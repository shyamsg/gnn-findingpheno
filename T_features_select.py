"""
Select features from the transcriptome data using different feature selection methods.
For now, we select the Transcriptome features independently from the Metagenome features.
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler

import torch.utils.data as Data


def normalize_data(data):
    sample_ids = data.index
    feature_names = data.columns
    data_matrix = data.values
    scaler = MinMaxScaler()
    scaled_matrix = scaler.fit_transform(data_matrix)
    scaled_data = pd.DataFrame(scaled_matrix, index=sample_ids, columns=feature_names)
    return scaled_data


# IMPORTANT NOTE, TODO: now the T features are selected to predict Pheno for the overall dataset, but we should make sure that they are equally good to predict Pheno for any of the samples, not just those in the main cluster
# TODO: TRY Multi-Modal autoencoder from MoGCN paper: 
import torch
from torch import nn
from matplotlib import pyplot as plt

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



SELECTION_METHOD = "MOGCN-VAE" # "Variance", "Autoencoder", "PCA", "Lasso", "MOGCN-VAE"

N_FEATURES_TO_USE = 50 # Only used for the Variance method
ALPHA = 0.01 # Only used for the Lasso method

def main():

    ### get the IDs of the sample with genomics data
    # sample_pcs = pd.read_excel("data/PCA/principalComponents_ofFish_basedOnGWAS.xlsx", header=0, index_col=0)    
    sample_pcs_index = (pd.read_csv("data/PCA/PCs_Fish_GWAS-based_cluster-filtered.csv", header=0, index_col=0)).index

    ### get the IDs of the sample with metagenomics and transcriptomics data
    # samples_with_MG_T_Ph_data = list((pd.read_csv("data/T_MG_P_input_data/final_input.csv", header=0, index_col=0)).index)
    # sample_index_final = sample_pcs_index[sample_pcs_index.isin(samples_with_MG_T_Ph_data)]
    # We removed the above lines: NOW WE SEPARATELY PROCESS MG and T! (see also clustering.py)


    ### METAGENOME features
    MG = pd.read_csv("data/HoloFish_MetaG_MeanCoverage_20221114.csv", header=0, index_col=0)
    MG = MG.loc[:, MG.columns.str.startswith('MAG')]

    ### TRANSCRIPTOME features (UNFILTERED)
    T_unfiltered = pd.read_csv("data/HoloFish_HostRNA_normalised_GeneCounts_230117.csv", header=0, index_col=0)
    T_unfiltered = T_unfiltered[T_unfiltered.index.isin(sample_pcs_index)]

    MG = MG.loc[MG.index.isin(T_unfiltered.index)]
    

    ### PHENOTYPE
    # MG_T_Ph_f = pd.read_csv("data/T_MG_P_input_data/final_input.csv", header=0, index_col=0)
    # Pheno = MG_T_Ph_f.loc[T_unfiltered.index, "weight"] # get the samples with metagenomics and transcriptomics data that are in the adjacency matrix
    # We removed the above lines: NOW WE PROCESS P SEPARETELY from MG and T! (see also clustering.py)

    Pheno = pd.read_csv("data/HoloFish_FishVariables_20221116.csv", header=0, index_col=0)
    Pheno = Pheno.loc[T_unfiltered.index, "Gutted.Weight.kg"]
    y = Pheno


    # # Calculate the average of each feature
    # T_feature_avg = T_unfiltered.mean(axis=0)
    # # print(T_feature_avg)


    if SELECTION_METHOD == "Variance":
        # Scale and normalize the features
        scaler = StandardScaler()
        T_scaled = scaler.fit_transform(T_unfiltered)


        # Convert the scaled features back to a DataFrame
        T_scaled_df = pd.DataFrame(T_scaled, columns=T_unfiltered.columns, index=T_unfiltered.index)

        # Save the scaled features to a CSV file
        T_scaled_df.to_csv("data/T_features/T_features_scaled.csv", index=True, sep=',')

        # Select the top N_FEATURES_TO_USE with the highest variance
        T_var_selected = T_unfiltered[T_unfiltered.var(axis=0).nlargest(N_FEATURES_TO_USE).index]
        file_name = "data/T_features/T_selected_features_" + SELECTION_METHOD + ".csv"
        T_var_selected.to_csv(file_name, index=True, sep=',')

        # Select the top N_FEATURES_TO_USE with the highest variance after scaling/normalizing
        T_var_selected_scaled = T_scaled_df[T_scaled_df.var(axis=0).nlargest(N_FEATURES_TO_USE).index]
        file_name = "data/T_features/T_selected_features_scaled_" + SELECTION_METHOD + ".csv"
        T_var_selected_scaled.to_csv(file_name, index=True, sep=',')

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


    # if SELECTION_METHOD == "PCA":

    # if SELECTION_METHOD == "Autoencoder":
        
    #     # Define the dimensions of the input data
    #     input_dim = T_unfiltered.shape[1]

    #     # Define the dimensions of the encoded representation
    #     ENCODING_DIM = 32

    #     # Define the input layer
    #     input_layer = Input(shape=(input_dim,))

    #     # Define the encoder layers
    #     encoder = Dense(ENCODING_DIM, activation='relu')(input_layer)

    #     # Define the decoder layers
    #     decoder = Dense(input_dim, activation='sigmoid')(encoder)

    #     # Define the autoencoder model
    #     autoencoder = Model(inputs=input_layer, outputs=decoder)

    #     # Compile the autoencoder model
    #     autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    #     # Train the autoencoder model
    #     autoencoder.fit(T_unfiltered, T_unfiltered, epochs=10, batch_size=32, shuffle=True)

    #     # Get the encoded features
    #     encoded_features = encoder.predict(T_unfiltered)

    #     # Convert the encoded features to a DataFrame
    #     encoded_features_df = pd.DataFrame(encoded_features, index=T_unfiltered.index)

    #     # Save the encoded features to a CSV file
    #     encoded_features_df.to_csv("data/T_features/T_features_encoded.csv", index=True, sep=',')


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
    

if __name__ == "__main__":
    main()