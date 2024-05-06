# Testing functions

import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

CLUSTERING_METHOS = "Hierarchical" # "K-means", "Hierarchical", "DBSCAN
MAX_POINTS_SELECTED_PER_CLUSTER = 5



def plot_dendrogram(Z, title="Hierarchical Clustering Dendrogram"):
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()


def main():

    sample_pcs = pd.read_excel("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/PCA/principalComponents_ofFish_basedOnGWAS.xlsx", header=0, index_col=0)    
    # sample_pcs = pd.read_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/PCA/PCs_Fish_GWAS-based_clustered_filtered.csv", header=0, index_col=0)


    # get the IDs of the sample with metagenomics and transcriptomics data
    samples_with_MG_T_Ph_data = list((pd.read_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/T_MG_P_input_data/final_input.csv", header=0, index_col=0)).index)

    sample_pcs = sample_pcs[sample_pcs.index.isin(samples_with_MG_T_Ph_data)]

    N_PCs = 200
    sample_pcs = sample_pcs.iloc[:,1:1 + N_PCs]
    # sample_pcs = sample_pcs.iloc[0:20, :]

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


        # breakpoint()
        plt.figure(figsize=(8, 6))
        plt.scatter(sample_pcs[1], sample_pcs[2])
        plt.scatter(centers[:,0], y=centers[:,1], color="red")


        plt.show()
        ########################################################################

    if CLUSTERING_METHOS == "Hierarchical":

        ### Hierarchical clustering ############
        ### To aggregate points that are very close to each other into their average, you can use hierarchical clustering to identify clusters of closely located points and then compute the average for each cluster
        
        # Perform hierarchical clustering
        Z = linkage(sample_pcs, method='average')  # method= 'ward', 'single', 'complete', 'average', 'weighted', 'centroid', 'median'

        plot_dendrogram(Z)
        
        # Determine clusters based on a distance threshold
        threshold = 900  # Adjust as needed
        clusters = fcluster(Z, threshold, criterion='distance')
        # Aggregate points within each cluster
        cluster_counts = np.bincount(clusters)[1:]
        print("Number of points in each cluster:\n", cluster_counts) # Number of points in each cluster

        cluster_averages = {}
        for cluster_id in np.unique(clusters):
            cluster_points = sample_pcs[clusters == cluster_id]
            cluster_average = np.mean(cluster_points, axis=0)
            cluster_averages[cluster_id] = cluster_average
        # Now cluster_averages contains the average point for each cluster

        
        unique_points = []
        for cluster_id in np.unique(clusters):
            cluster_points = sample_pcs[clusters == cluster_id]
            for i in range(0, min(cluster_points.shape[0], MAX_POINTS_SELECTED_PER_CLUSTER)): # Keep at most "MAX_..." points per cluster
                unique_points.append(cluster_points.iloc[i]) 
        # TODO: Can we IMPROVED: instead of keeping the first "MAX_..." points, we can keep the points that are FURTHEST from each other in the cluster

        sample_pcs_d = pd.DataFrame(unique_points)

        sample_pcs_d.to_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/PCA/PCs_Fish_GWAS-based_clustered_filtered.csv", index=True, sep=',')

        Z = linkage(sample_pcs_d, method='average')
        plot_dendrogram(Z, title="Hierarchical Clustering Dendrogram after selecting at most {} points per cluster".format(MAX_POINTS_SELECTED_PER_CLUSTER))

        # We can repeat this process iteratively to further reduce the number of points. We get to a point where each cluster contains exactly 1 point except for one cluster, which contains the remaining points. We keep only one point for this cluster. Then repeat.
        # breakpoint()


    # TODO
    # if CLUSTERING_METHOS == "DBSCAN":

if __name__ == "__main__":
    main()