"""
Clustering based on gene distances
Output a pcs file with the selected points from each cluster
"""


import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial import distance
from scipy.spatial.distance import pdist
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


N_PCs = 8  #TODO MAKE THIS A USED DEFINED PARAMETER, it could also be a percentage of the variance explained. IT COULD ALSO be optimized using optuna
# Try just 2 PCs to avoid overfitting, which explains 0.12687 of the phenotypic variance

CLUSTERING_METHOS = "Hierarchical" # "K-means", "Hierarchical", todo:"DBSCAN
MAX_POINTS_SELECTED_PER_CLUSTER = 500 # Must be >=1


def plot_dendrogram(Z, title="Hierarchical Clustering Dendrogram"):
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()


def main():

    sample_pcs = pd.read_excel("data/PCA/principalComponents_ofFish_basedOnGWAS.xlsx", header=0, index_col=0)    
    # sample_pcs = pd.read_csv("data/PCA/PCs_Fish_GWAS-based_cluster-filtered.csv", header=0, index_col=0)

    # get the IDs of the sample with metagenomics and transcriptomics data
    # WE SEPARATELY PROCESS MG and T! samples_with_MG_T_Ph_data = list((pd.read_csv("data/T_MG_P_input_data/final_input.csv", header=0, index_col=0)).index)
    T_samples = pd.read_csv("data/HoloFish_HostRNA_normalised_GeneCounts_230117.csv", header=0, index_col=0)
    T_index = list(T_samples.index)
    MG_samples = pd.read_csv("data/HoloFish_MetaG_MeanCoverage_20221114.csv", header=0, index_col=0)
    MG_index = list(MG_samples.index)

    # T_MG_samples = pd.read_csv("data/kegg/scaled_input_carbo.csv", header=0, index_col=0) # scaled_input_fatty.csv, scaled_input_carbo.csv, scaled_input_lipid.csv
    # T_index = MG_index = T_MG_samples.index

    
    samples_ids_T_MG = list(set(T_index).intersection(set(MG_index)))
    # TODO: do union instead of intersection: Add the samples that only have G (+either T or MG) data to the graph. Input missing data (MG or T) for the samples that are missing such data by averaging the neighbors(Synthetic data)
    samples_ids_T_MG = list(set(T_index).union(set(MG_index))) # We get 453 instead of 288 samples, and 354 instead of 244 samples when intersecting with G data

    ### IMPORTANT: we have 40 sample for which there are both MG and T but no G: TODO: can we add G in some way? (average the neighbors?)
    missing_samples = [sample for sample in samples_ids_T_MG if sample not in sample_pcs.index]
    print("Samples without G:", len(missing_samples))

    sample_pcs = sample_pcs[sample_pcs.index.isin(samples_ids_T_MG)]
    print("Shape of sample_pcs (sample x PC matrix):", sample_pcs.shape)

    sample_pcs = sample_pcs.iloc[:,0:N_PCs]
    sample_pcs.columns = ["PC" + str(col) for col in sample_pcs.columns]

    if CLUSTERING_METHOS == "K-means":
        ### K-means clustering ############
        # given some points in a N dimensional space, cluster them into K components, where each component is the average of the group

        K = 3

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=K)
        kmeans.fit(sample_pcs)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_ 

        # center = np.column_stack((centers, ["red"]*K))

        print("Centers: \n", centers)


        plt.figure(figsize=(8, 6))
        plt.scatter(sample_pcs[1], sample_pcs[2])
        plt.scatter(centers[:,0], y=centers[:,1], color="red")


        plt.show()
        ########################################################################

    if CLUSTERING_METHOS == "Hierarchical":

        THRESHOLD = 120 # Adjust as needed

        ### Hierarchical clustering ### To aggregate points that are very close to each other into their average, you can use hierarchical clustering to identify clusters of closely located points and then compute the average for each cluster
        
        # Perform hierarchical clustering
        def perform_hc(sample_pcs, threshold, method='average', plot=True, title_plot="Hierarchical Clustering Dendrogram"):
            Z = linkage(sample_pcs, method=method)  # method= 'ward', 'single', 'complete', 'average', 'weighted', 'centroid', 'median'
            # if plot: plot_dendrogram(Z, title=title_plot)
            clusters = fcluster(Z, threshold, criterion='distance') # Determine clusters based on a distance THRESHOLD
            cluster_counts = np.bincount(clusters)[1:] # Aggregate points within each cluster
            print("Number of points in each cluster:\n", cluster_counts) # Number of points in each cluster
            print("Number of clusters: ", len(cluster_counts))
            return clusters

        clusters = perform_hc(sample_pcs, THRESHOLD)

        cluster_averages = {} # Compute the average of each cluster
        for cluster_id in np.unique(clusters):
            cluster_averages[cluster_id] = np.mean(sample_pcs[clusters == cluster_id], axis=0)


        
        def select_furthest_points(sample_pcs, clusters, max_points_per_cluster): # Select at most "MAX_POINTS_SELECTED_PER_CLUSTER" points per cluster
            selected_points = []
            for cluster_id in np.unique(clusters):
                cluster_points = sample_pcs[clusters == cluster_id]
            # Keep at most "MAX_POINTS_SELECTED_PER_CLUSTER" points per cluster
                # for i in range(0, min(cluster_points.shape[0], MAX_POINTS_SELECTED_PER_CLUSTER)):
                #     selected_points.extend(cluster_points.iloc[i].index.tolist()) 
                # IMPROVEMENT: instead of keeping the first "MAX_POINTS_SELECTED_PER_CLUSTER" points, we keep the points of each cluster that are FURTHEST from each other in the cluster
                num_points = cluster_points.shape[0]
                if num_points <= max_points_per_cluster:
                    selected_points.extend(cluster_points.index.tolist())
                else:
                    new_cluster_points = []
                    # for i in range(0, MAX_POINTS_SELECTED_PER_CLUSTER-1):
                    while len(new_cluster_points) < max_points_per_cluster:
                        distances = pdist(cluster_points)
                        dist_matrix = distance.squareform(distances)
                        max_distance_indices = list(np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape))
                        furthest_points = cluster_points.iloc[max_distance_indices].index.tolist()
                        new_cluster_points.extend(furthest_points)
                        new_cluster_points = list(set(new_cluster_points))
                        cluster_points = cluster_points.drop(furthest_points[1])
                    selected_points.extend(new_cluster_points)
            return list(set(selected_points))

        selected_points = select_furthest_points(sample_pcs, clusters, MAX_POINTS_SELECTED_PER_CLUSTER)
        sample_pcs_d = (sample_pcs.loc[selected_points]).sort_index()

        # Perform hierarchical clustering again
        clusters = perform_hc(sample_pcs_d, THRESHOLD, title_plot= "Hierarchical Clustering Dendrogram after selecting at most the furthest {} points per cluster".format(MAX_POINTS_SELECTED_PER_CLUSTER), plot=True)
        
        # We can repeat this process iteratively to further reduce the number of points. We get to a point where each cluster contains exactly 1 point except for one cluster, which contains the remaining points. We keep only one point for this cluster. Then repeat.

        sample_pcs_d['cluster_id'] = clusters
        print("\nsaving OUTPUT: data/PCA/PCs_Fish_GWAS-based_cluster-filtered.csv")
        sample_pcs_d.to_csv("data/PCA/PCs_Fish_GWAS-based_cluster-filtered.csv", index=True, sep=',')

        samples_ids = sample_pcs_d.index.tolist()

        Pheno = pd.read_csv("data/HoloFish_FishVariables_20221116.csv", header=0, index_col=0)
        Pheno = Pheno.loc[samples_ids, "Gutted.Weight.kg"]


        # Visualization (and R2 correlation) of the selected points
        samples_pheno = pd.concat([sample_pcs_d.iloc[:,0:N_PCs], Pheno], axis=1)

        print("saving OUTPUT: data/PCs_Fish_GWAS-based_cluster-filtered_withPheno.csv")
        samples_pheno.to_csv("data/PCs_Fish_GWAS-based_cluster-filtered_withPheno.csv", index=True, sep=',')

        samples_pheno_matrix = samples_pheno.to_numpy()
        pc1_values, pc2_values = samples_pheno_matrix[:, 0], samples_pheno_matrix[:, 1]

        plt.figure(figsize=(10, 6))
        # plt.scatter(pc1_values, pheno_values, alpha=0.7) # OLD ONE WITH JUST ONE PRINCIPAL COMPONENT
        plt.xlabel("PC1")
        #plt.ylabel("Weight") # if 1 dimension
        plt.ylabel("PC2")
        plt.scatter(pc1_values, pc2_values, c=Pheno, cmap='viridis', alpha=0.7)
        plt.colorbar(label='Weight')
        plt.title("Samples Plotted on PC1,PC2 x Weight")
        plt.show()

        ### Calculate R2 correlation value between PC values and weight
        X = samples_pheno_matrix[:, :N_PCs]  # PC1 and PC2 values, the last column is the weight!! You need to exclude it here
        y = Pheno #samples_pheno_matrix[:, -1]  # Weight values

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(X, y)
        # Predict the weight values
        y_pred = model.predict(X)
        # Calculate the R2 score
        r2 = r2_score(y, y_pred)
        print("\nR2 correlation value between PC values and weight:", r2)
        
        print("Final shape:", sample_pcs_d.shape) # There is one additional column because we added the cluster_id column, that's why the shape is (samplex x PCs + 1)

    # TODO DBSCAN
    # if CLUSTERING_METHOS == "DBSCAN":

if __name__ == "__main__":
    main()