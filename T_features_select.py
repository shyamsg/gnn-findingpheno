
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

SELECTION_METHOD = "Variance" # "Variance", "Autoencoder", "PCA"

N_features_to_use = 100

def main():

    sample_pcs = pd.read_excel("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/PCA/principalComponents_ofFish_basedOnGWAS.xlsx", header=0, index_col=0)    
    # sample_pcs = pd.read_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/PCA/PCs_Fish_GWAS-based_cluster-filtered.csv", header=0, index_col=0)

    # get the IDs of the sample with metagenomics and transcriptomics data
    samples_with_MG_T_Ph_data = list((pd.read_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/T_MG_P_input_data/final_input.csv", header=0, index_col=0)).index)

    sample_pcs = sample_pcs[sample_pcs.index.isin(samples_with_MG_T_Ph_data)]

    # Read the T features (UNFILTERED)
    T_unfiltered = pd.read_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/HoloFish_HostRNA_normalised_GeneCounts_230117.csv", header=0, index_col=0)
    T_unfiltered = T_unfiltered[T_unfiltered.index.isin(sample_pcs.index)]

    # Calculate the average of each feature
    T_feature_avg = T_unfiltered.mean(axis=0)
    # print(T_feature_avg)


    if SELECTION_METHOD == "Variance":
        # Scale and normalize the features
        scaler = StandardScaler()
        T_scaled = scaler.fit_transform(T_unfiltered)


        # Convert the scaled features back to a DataFrame
        T_scaled_df = pd.DataFrame(T_scaled, columns=T_unfiltered.columns, index=T_unfiltered.index)

        # Save the scaled features to a CSV file
        T_scaled_df.to_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/T_features_scaled.csv", index=True, sep=',')

        # Select the top N_features_to_use with the highest variance
        T_var_selected = T_unfiltered[T_unfiltered.var(axis=0).nlargest(N_features_to_use).index]    
        T_var_selected.to_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/T_features_var-selected.csv", index=True, sep=',')

        # Select the top N_features_to_use with the highest variance after scaling/normalizing
        T_var_selected_scaled = T_scaled_df[T_scaled_df.var(axis=0).nlargest(N_features_to_use).index]
        T_var_selected_scaled.to_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/T_features_var-selected.csv", index=True, sep=',')



if __name__ == "__main__":
    main()