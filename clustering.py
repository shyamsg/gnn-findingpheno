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


CLUSTERING_METHOS = "Hierarchical" # "K-means", "Hierarchical", "DBSCAN
MAX_POINTS_SELECTED_PER_CLUSTER = 5 # Must be >=1


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
    T_samples = list((pd.read_csv("data/HoloFish_HostRNA_normalised_GeneCounts_230117.csv", header=0, index_col=0)).index)
    MG_samples = list((pd.read_csv("data/HoloFish_MetaG_MeanCoverage_20221114.csv", header=0, index_col=0)).index)

    sample_pcs = sample_pcs[sample_pcs.index.isin(T_samples) & sample_pcs.index.isin(MG_samples)]

    print("Shape of sample_pcs (samplex x PCs):", sample_pcs.shape)

    # N_PCs = sample_pcs.shape[1] # We have more PCs than samples. PCA was run on the original 361 samples (all samples with GWAS data), but we have fewer samples that have both metagenomics and transcriptomics data
    N_PCs = 132

    sample_pcs = sample_pcs.iloc[:,0:N_PCs]

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


        plt.figure(figsize=(8, 6))
        plt.scatter(sample_pcs[1], sample_pcs[2])
        plt.scatter(centers[:,0], y=centers[:,1], color="red")


        plt.show()
        ########################################################################

    if CLUSTERING_METHOS == "Hierarchical":

        THRESHOLD = 500 # Adjust as needed

        ### Hierarchical clustering ############
        ### To aggregate points that are very close to each other into their average, you can use hierarchical clustering to identify clusters of closely located points and then compute the average for each cluster
        
        # Perform hierarchical clustering
        Z = linkage(sample_pcs, method='average')  # method= 'ward', 'single', 'complete', 'average', 'weighted', 'centroid', 'median'

        plot_dendrogram(Z)
        
        # Determine clusters based on a distance THRESHOLD
        clusters = fcluster(Z, THRESHOLD, criterion='distance')
        # Aggregate points within each cluster
        cluster_counts = np.bincount(clusters)[1:]
        print("Number of points in each cluster:\n", cluster_counts) # Number of points in each cluster
        print("Number of clusters: ", len(cluster_counts))

        cluster_averages = {}
        for cluster_id in np.unique(clusters):
            cluster_points = sample_pcs[clusters == cluster_id]
            cluster_average = np.mean(cluster_points, axis=0)
            cluster_averages[cluster_id] = cluster_average
        # Now cluster_averages contains the average point for each cluster


        selected_points = []
        for cluster_id in np.unique(clusters):
            cluster_points = sample_pcs[clusters == cluster_id]
        # Keep at most "MAX_POINTS_SELECTED_PER_CLUSTER" points per cluster
            # for i in range(0, min(cluster_points.shape[0], MAX_POINTS_SELECTED_PER_CLUSTER)):
            #     selected_points.extend(cluster_points.iloc[i].index.tolist()) 
        # IMPROVEMENT: instead of keeping the first "MAX_POINTS_SELECTED_PER_CLUSTER" points, we keep the points of each cluster that are FURTHEST from each other in the cluster
            num_points = cluster_points.shape[0]
            if num_points <= MAX_POINTS_SELECTED_PER_CLUSTER:
                selected_points.extend(cluster_points.index.tolist())
            else:
                new_cluster_points = []
                # for i in range(0, MAX_POINTS_SELECTED_PER_CLUSTER-1):
                while len(new_cluster_points) < MAX_POINTS_SELECTED_PER_CLUSTER:
                    distances = pdist(cluster_points)
                    dist_matrix = distance.squareform(distances)
                    max_distance_indices = list(np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape))

                    furthest_points = cluster_points.iloc[max_distance_indices].index.tolist()
                    new_cluster_points.extend(furthest_points)
                    new_cluster_points = list(set(new_cluster_points))
                    cluster_points = cluster_points.drop(furthest_points[1]) # 1 or 0. We drop one of the furthest points from the list, so that we can find the next furthest point
                # breakpoint()
                selected_points.extend(new_cluster_points)
        selected_points = list(set(selected_points))
        sample_pcs_d = sample_pcs.loc[selected_points]

        # sample_pcs_d = pd.DataFrame(selected_points)
        sample_pcs_d = sample_pcs_d.sort_index()

        Z = linkage(sample_pcs_d, method='average')
        plot_dendrogram(Z, title="Hierarchical Clustering Dendrogram after selecting at most the furthest {} points per cluster".format(MAX_POINTS_SELECTED_PER_CLUSTER))
        clusters = fcluster(Z, THRESHOLD, criterion='distance')
        cluster_counts = np.bincount(clusters)[1:]
        print("Number of points in each cluster:\n", cluster_counts) # Number of points in each cluster
        print("Number of clusters: ", len(cluster_counts))
        

        sample_pcs_d['cluster_id'] = clusters
        sample_pcs_d.to_csv("data/PCA/PCs_Fish_GWAS-based_cluster-filtered.csv", index=True, sep=',')

        samples_ids = sample_pcs_d.index.tolist()

        Pheno = pd.read_csv("data/HoloFish_FishVariables_20221116.csv", header=0, index_col=0)
        Pheno = Pheno.loc[samples_ids, "Gutted.Weight.kg"]



        # !!! We take just the first 200 PCs for the sake of visualization
        samples_pheno = pd.concat([sample_pcs_d.iloc[:,0:200], Pheno], axis=1)
        samples_pheno.to_csv("data/PCs_Fish_GWAS-based_cluster-filtered_withPheno.csv", index=True, sep=',')


        # Plot the samples on the first principal component axis

        samples_pheno_matrix = samples_pheno.to_numpy()
        pc1_values = samples_pheno_matrix[:, 0]
        pc2_values = samples_pheno_matrix[:, 1]
        pheno_values = samples_pheno_matrix[:, -1]

        # Plot each sample on the x-axis using PC1 values
        plt.figure(figsize=(10, 6))
        # plt.scatter(pc1_values, np.zeros_like(pc1_values), alpha=0.7)
        # plt.scatter(pc1_values, pheno_values, alpha=0.7) # OLD ONE WITH JUST ONE PRINCIPAL COMPONENT
        plt.xlabel("PC1")
        #plt.ylabel("Weight")
        plt.ylabel("PC2")
        plt.scatter(pc1_values, pc2_values, c=pheno_values, cmap='viridis', alpha=0.7)
        plt.colorbar(label='Weight')
        # plt.yticks([])  # Hide y-axis since all points are at zero on y
        plt.title("Samples Plotted on PC1,PC2 x Weight")
        plt.show()


        
        # Calculate R2 correlation value between PC values and weight
        X = samples_pheno_matrix[:, :132]  # PC1 and PC2 values, the last column is the weight!! You need to exclude it here
        y = samples_pheno_matrix[:, -1]  # Weight values

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict the weight values
        y_pred = model.predict(X)

        # Calculate the R2 score
        r2 = r2_score(y, y_pred)
        print("R2 correlation value between PC values and weight:", r2)
        
        # We can repeat this process iteratively to further reduce the number of points. We get to a point where each cluster contains exactly 1 point except for one cluster, which contains the remaining points. We keep only one point for this cluster. Then repeat.

        print("Final shape:", sample_pcs_d.shape) # There is one additional column because we added the cluster_id column, that's why the shape is (samplex x PCs + 1)



    # TODO DBSCAN
    # if CLUSTERING_METHOS == "DBSCAN":

if __name__ == "__main__":
    main()