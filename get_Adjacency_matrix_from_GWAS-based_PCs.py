from torch_geometric.data import Data
import numpy as np
import pandas as pd
from scipy.spatial import distance
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
import matplotlib.pyplot as plt

from similarity_graph_utilities import get_edges_from_adjacency
from similarity_graph_utilities import plot_gr


BINARY_EDGES = False


def get_similarity_matrix(sample_pcs, n_samples, print_tmp=False):
    """
    Get n_sample x n_sample similarity matrix from n_sample x n_PCs matrix
    Eucledian distance: based on normal distribution. Natural connection between Gaussian kernels and Euclinead distance

    Parameters:
    - parameter1 (type): Description of parameter1
    
    Returns:
    - return: Sample_similarity_matrix
    """

    #TODO Shyam will send a new distance matrix (start from this, it is not from PCA) to compute the similarity matrix from. We will try different distance matrices, different wats to build the distance matrix
    #TODO: Compute the weighted distance using eigenvalues. Note: this should be done already in the implementation of the PCA
    #TODO: CHECK the TRACE ( trace(xx') == trace(xstar xstar') )

    # Compute the pairwise distances between points
    pairwise_dist = distance.pdist(sample_pcs, 'euclidean')
    if print_tmp: print("\nPairwise Distance:\n", pairwise_dist)
    # Convert the distances to a square matrix
    dist_matrix = distance.squareform(pairwise_dist)
    if print_tmp: print("\nPairwise Distance Matrix:\n", dist_matrix)
    
    # Keep the indexes of the matrix. FOR NOW, WE ADD THEM AT THE END
    # indexes = np.triu_indices(dist_matrix.shape[0], k=1)
    # pairwise_dist_with_indexes = dist_matrix[indexes]


    # Normalize the distance matrix on the maximum distance
    max_distance=np.max(dist_matrix)
    if print_tmp: print("\nmax distance between any 2 points: ", max_distance)
    dist_matrix/=max_distance
    if print_tmp: print("\nMatrix after normalization\n",dist_matrix)

    # Compute the similarity matrix as 1 - distance
    # sample_similarity_matrix = dist_matrix
    sample_similarity_matrix = np.ones((n_samples,n_samples))-dist_matrix
    # TODO INSTEAD OF FOCUSING ON SIMILARITY, USE THE DISTANCES: sample_similarity_matrix = dist_matrix
    # We want 0s insted of 1s for the diagonal
    np.fill_diagonal(sample_similarity_matrix, 0)
    if print_tmp: print("\nMatrix after 1-dist_matrix and 0s in the diagonal operation\n",sample_similarity_matrix)

    df = pd.DataFrame(sample_similarity_matrix)
    df.to_csv("data/PCA/sample_similarity_matrix.tsv", sep='\t', index=True, header=True)

    return sample_similarity_matrix


def analyze_similarity(upper_diag_vector):
    #### plot a histogram of similarity-based edges distribution in order to choose a cutoff value
    num_bins = 100
    bin_edges = np.linspace(0.0, 1.0, num_bins+1)
    thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.8, 0.9]
    for threshold in thresholds: print(f">{threshold}:\n", sum(upper_diag_vector > threshold)) # PRINT NUMBER OF SIMILARITY-BASED EDGES FOR A GIVEN THRESHOLD VALUE
    plot_sample_similarity_distribution(upper_diag_vector, num_bins)


def plot_sample_similarity_distribution(upper_diag_vector, num_bins):
    """
    Parameters:
    - parameter1 (type): Description of parameter1.

    """

    # Create a histogram plot for visualization
    plt.hist(upper_diag_vector, bins=num_bins+1, edgecolor='black')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Discretized Continuous Data')
    plt.grid(True)
    plt.show()

    return


def select_edges(sample_similarity_matrix, indexes, cluster_id, cutoff, min_edges=12, max_edges=25, print_tmp=False, binary_edges=True):
    """
    Filter the similarity edges based on 3 criteria:
    1 - Similarity value above cutoff
    2 - Min (min_edges) connections for each node
    3 - Max (max_edges) connections for each node

    FINAL MATRIX = (M1 | M2) & M3

    Parameters:
    - sample_similarity_matrix (numpy.ndarray): matrix n_samples x n_samples whose values represent similarities.
    - cutoff
    
    Returns:
    - filtered_similarity_matrix
    """

    MAX_CLUSTER_DIM = 10

    # save_path = "data/tmp_bfr.csv"
    # df = pd.DataFrame(sample_similarity_matrix)
    # df.to_csv(save_path, sep=',', index=True, header=True)

        
    if binary_edges == True: # BINARY VALUES #
        ### ADDED: PRE-FILTER within CLUSTERs (FILTER EDGES BASED ON WITHIN-CLUSTER CONNECTIVITY LIMIT) # TODO ADD THIS TO A SEPARATE FUNCTION AND CALL IT ONCE (INSTEAD OF FOR EACH MIN_EDGES, MAX_EDGES COMBINATION)
        indexes_cluster_only = [index.split("_")[1] for index in indexes]

        unique_cluster_ids, counts = np.unique(indexes_cluster_only, return_counts=True)
        clusters_to_reduce = unique_cluster_ids[np.where(counts > MAX_CLUSTER_DIM)].astype(int)
        print("Clusters to reduce: ", clusters_to_reduce)


        MAX_EDGES_WITHIN_CLUSTERS = 1 # we keep only the top connection for each node within the (highly connected) cluster


        for id_cluster in clusters_to_reduce:
            cluster_indices = np.where(cluster_id == id_cluster)[0]
            cluster = sample_similarity_matrix[cluster_indices][:, cluster_indices]

            if print_tmp: 
                print("\nCluster ", id_cluster)
                print("BEFORE WITHIN-CLUSTER FILTERING")
                print("Total number of edges for the given cutoff value:", sum(sum(cluster>0)))
                print("number of within-cluster edges for each node in the cluster:\n", np.sum(cluster>cutoff, axis=1))

            # number of edges above the cutoff

            # Find the indices of the N highest values in each row
            top_indices = np.argsort(cluster, axis=1)[:, -MAX_EDGES_WITHIN_CLUSTERS:]
            # Create a mask to set the top N values to 1 and others to 0
            rows = np.arange(cluster.shape[0])[:, None]
            m_cluster = np.zeros_like(cluster)
            m_cluster[rows, top_indices] = 1
            m_cluster[top_indices, rows] = 1 # ! Added this line to ensure symmetry (if you add an edge from i to j, also add from j to i)
            m_cluster = m_cluster.astype(float)
            # print(np.allclose(m3,m3.T)) # check that the matrix is symmetric

            for num_r,i in enumerate(cluster_indices):
                for num_c,j in enumerate(cluster_indices):
                    sample_similarity_matrix[i,j] = m_cluster[num_r,num_c] # !!! we update the sample_similarity_matrix with the cluster filter

            if print_tmp:
                print("AFTER WITHIN-CLUSTER FILTERING")
                print("Total number of edges for the given cutoff value:", sum(sum(m_cluster>0)))
                print("number of within-cluster edges for each node in the cluster:\n",np.sum(m_cluster, axis=1).astype(int))     



        ##### Three matrices to merge:

        #### 1: Keep the edges above the cutoff
        m1 = (sample_similarity_matrix>cutoff).astype(int) # numpy.ndarray, (361,361)
        if print_tmp: print("m1:\n", m1)

        
        #### 2: Keep the top connection for each node

        # Find the indices of the N highest values in each row
        top_indices = np.argsort(sample_similarity_matrix, axis=1)[:, -min_edges:]
        # Create a mask to set the top N values to 1 and others to 0
        m2 = np.zeros_like(sample_similarity_matrix)
        rows = np.arange(sample_similarity_matrix.shape[0])[:, None]
        m2[rows, top_indices] = 1
        m2[top_indices, rows] = 1 # ! Added this line to ensure symmetry (if you add an edge from i to j, also add from j to i)
        m2 = m2.astype(float)
        # print(np.allclose(m2,m2.T)) # check that the matrix is symmetric

        row_sums = np.sum(m2, axis=1)
        # for i, row_sum in enumerate(row_sums):
        #    print("Sum of row {}: {}".format(i+1, row_sum)) # OUTPUT should be >= min_edges for each line


        # max_indices = np.argmax(sample_similarity_matrix, axis=1)
        # m2 = np.zeros_like(sample_similarity_matrix)
        # m2[np.arange(m2.shape[0]), max_indices] = 1 # set to 1 the maximum value in each row
        # m2[max_indices, np.arange(m2.shape[1])] = 1 # ! Added this line to ensure symmetry (if you add an edge from i to j, also add from j to i)
        # print(np.allclose(m2,m2.T)) # check that the matrix is symmetric

        if print_tmp: print("\nm2:\n", m2)


        #### 3: Take the top N edges for each node (maximum number of edges for each node)
        # Find the indices of the N highest values in each row
        top_indices = np.argsort(sample_similarity_matrix, axis=1)[:, -max_edges:]
        # Create a mask to set the top N values to 1 and others to 0
        m3 = np.zeros_like(sample_similarity_matrix)
        rows = np.arange(sample_similarity_matrix.shape[0])[:, None]
        m3[rows, top_indices] = 1
        m3[top_indices, rows] = 1 # ! Added this line to ensure symmetry (if you add an edge from i to j, also add from j to i)
        m3 = m3.astype(float)
        # print(np.allclose(m3,m3.T)) # check that the matrix is symmetric

        if print_tmp: print("\nm3:\n", m3)
        if print_tmp: print("\nNumber of connections per node:\n",np.sum(m3, axis=1)) # OUTPUT should be >=max_edges for each line (it can be >20 because after considering the top 20 edges for each node, we add the symmetric)


        ##### Union between the 3 matrices
        filtered_similarity_matrix = np.logical_or(m1, m2).astype(float)
        if print_tmp: print("m1+m2:\n",sum(filtered_similarity_matrix>0))
        # print(np.sum(filtered_similarity_matrix, axis=1)) # OUTPUT should be =20 for each line # OUTPUT should be >=1 for each line
        filtered_similarity_matrix *= m3
        # print(np.sum(filtered_similarity_matrix, axis=1)) # OUTPUT should be >=1 AND <=20 for each line
        #np.set_printoptions(threshold=np.inf) # print the full matrix instead of the truncated version
        if print_tmp: print("\nfinal matrix\n",filtered_similarity_matrix)
    
    
    else: # REAL VALUES (edge weights are not binarized)
        sample_similarity_matrix[sample_similarity_matrix < cutoff] = 0 # Keep the edges above the cutoff
        filtered_similarity_matrix = sample_similarity_matrix
        # filtered_similarity_matrix = (sample_similarity_matrix>cutoff).astype(float)
        if print_tmp: print("filtered_similarity_matrix:\n", filtered_similarity_matrix)

        # save_path = "data/tmp.csv"
        # df = pd.DataFrame(filtered_similarity_matrix)
        # df.to_csv(save_path, sep=',', index=True, header=True)
      

    # print("Number of edges: ", np.sum(filtered_similarity_matrix))
    # num_filtered_edges = np.sum((filtered_similarity_matrix > 0) & (filtered_similarity_matrix < cutoff)) # CHECK THAT THIS IS EQUAL TO 0

    return filtered_similarity_matrix


def filter(adj_matrix):
    """
    
    Parameters:
    - adj_matrix (numpy.ndarray): Adjacency matrix representing the graph.
    """
    # Get the indices of the 10 nodes with the most edges
    top_10_nodes = np.argsort(np.sum(adj_matrix, axis=1))[-10:]
    
    # Create a subgraph with only the top 10 nodes and their edges
    subgraph = adj_matrix[top_10_nodes][:, top_10_nodes]
    
    # Plot the subgraph
    plt.imshow(subgraph, cmap='binary')
    plt.colorbar()
    plt.title('Graph with Top 10 Nodes')
    plt.show()



N_PCs = 360 # Defined after using the R script (see google colab)


def main():

    # sample_pcs = pd.read_excel("data/PCA/principalComponents_ofFish_basedOnGWAS.xlsx", header=0, index_col=0)
    sample_pcs = pd.read_csv("data/PCA/PCs_Fish_GWAS-based_cluster-filtered.csv", header=0, index_col=0)
    # print(sample_pcs)

    # COUNT THE NUMBER OF SAMPLES PER EACH CLUSTER
    sample_pcs = pd.DataFrame(sample_pcs)
    counts_groupby_size = sample_pcs.groupby('cluster_id').size().reset_index(name='count')
    counts_groupby_size = counts_groupby_size[counts_groupby_size['count'] > 3]
    # print(counts_groupby_size)

    ## removing a specific sample
    # row_F400 = sample_pcs.loc["F400"]
    # sample_pcs = sample_pcs.drop("F365")



    ### FILTERING for MG T data availability ###!!! WE NO LONGER DO THIS HERE, ALL THE SAMPLES WITH MG, T (and Ph) DATA HAVE BEEN ALREADY SELECTED (see clustering.py)
    # MG_T_Ph = pd.read_csv("data/T_MG_P_input_data/final_input.csv", header=0, index_col=0)
    # # get the IDs of the sample with metagenomics and transcriptomics data
    # samples_with_MG_T_Ph_data = list(MG_T_Ph.index)
    # ## Keep only the samples for which we have transcriptomics and metagenomics data. 'final_input' is the file that contains all transcriptomics and metagenomics data for the samples we have such data for.
    # sample_pcs = sample_pcs[sample_pcs.index.isin(samples_with_MG_T_Ph_data)]

    # print("\n\n\n", sample_pcs.index)
    # print(len(sample_pcs.index))

    N_SAMPLES= sample_pcs.shape[0]
    # print(N_SAMPLES) # 207 after filtering, 361 before filtering (now we added cluster-filtering, so the number of samples is different from the one in the R script)

    cluster_id = sample_pcs["cluster_id"]
    
    # We select the first N_PCs features. N_PCs = 50 account for ~50% variability (see R file for the PC analysis)
    sample_pcs = sample_pcs.iloc[:,0:N_PCs]
    # sample_pcs = pd.concat([sample_pcs.iloc[:,0:N_PCs], sample_pcs["cluster_id"]], axis=1) # we add the cluster_id to the PCs, to limit the within-cluster connectivity
    
    indexes = sample_pcs.index
    sample_pcs.index = sample_pcs.index + "_" + cluster_id.astype(str) # we add the cluster_id to the indexes of the samples
    indexes_with_clusters = sample_pcs.index

    sample_similarity_matrix = get_similarity_matrix(sample_pcs, n_samples=N_SAMPLES, print_tmp=False)
    print("\nSample_similarity_matrix:\n", sample_similarity_matrix)
    # print("SHAPE:", sample_similarity_matrix.shape) # (361,361)
    # print("SHAPE:",sample_similarity_matrix.flatten() > 0.4)

    # sample_similarity_matrix = pd.DataFrame(sample_similarity_matrix, index=indexes_with_clusters, columns=indexes_with_clusters)
    
    # # sample_similarity_matrix_c = sample_similarity_matrix.loc[sample_similarity_matrix.index.str.contains("_84")]
    # sample_similarity_matrix_c = sample_similarity_matrix.loc[:,sample_similarity_matrix.columns.str.contains("_84")]



    # def keep_max_only(row):
    #     # Create a boolean mask where the max value(s) will be True
    #     mask = row == row.max()
    #     # Zero out values that are not the max
    #     row[~mask] = 0
    #     return row

    # # Apply the function to each row
    # sample_similarity_matrix_c_2 = sample_similarity_matrix_c.apply(keep_max_only, axis=1)

    # print(sample_similarity_matrix_c_2["F365_84"])
    # for i in sample_similarity_matrix_c_2["F365_84"]:print(i)


    rows, cols = sample_similarity_matrix.shape
    # Extract the upper diagonal elements into a vector
    upper_diag_vector = sample_similarity_matrix[np.triu_indices(rows, k=1)]
    analyze_similarity(upper_diag_vector) # UNCOMMENT to plot a histogram of similarity-based edges distribution in order to choose a cutoff value
    CUTOFF = 0.20 # we choose a cutoff value based on the histogram of the similarity-based edges distribution


    print("cutoff matrix:\n",sum(sample_similarity_matrix > CUTOFF)) # shape will be (n_samples,)
    
    if BINARY_EDGES:
        for MIN_EDGES in 1,3,4,5,6,7,8,9,10,11,12,13,14,15,17,20,25,30:
            for MAX_EDGES in 5,6,8,10,12,14,17,20,23,26,30,35,40,50:
                if MAX_EDGES > MIN_EDGES:


                    adjacency_matrix = select_edges(sample_similarity_matrix, indexes_with_clusters, cluster_id, cutoff=CUTOFF, min_edges=MIN_EDGES, max_edges=MAX_EDGES, print_tmp=False, binary_edges=True)
                    # print("\nFiltered similarity matrix:\n", adjacency_matrix)
                    
                    print("Min edges: ", str(MIN_EDGES), " - Max edges: ", str(MAX_EDGES), "Number of edges: ", np.sum(adjacency_matrix))

                    row_sums = np.sum(adjacency_matrix, axis=1)
                    # print("\nNumber of edges per each node after filtering:\n", row_sums)
                    
                    # ### KEEP ONLY N% OF THE EDGES IN THE CLUSTER 77, which is the largest cluster, composed of 54 highly similar (-> high connected) samples
                    # ID_CLUSTER = 77
                    # # Extract the submatrix
                    # cluster_indices = np.where(cluster_id == ID_CLUSTER)[0]
                    # cluster = adjacency_matrix[cluster_indices][:, cluster_indices]    
                    # # Find the indices of all 1s in the submatrix
                    # ones_indices = np.argwhere(cluster == 1)
        
                    # # Calculate the number of 1s to change (20% of the total 1s)
                    # num_ones = len(ones_indices)
                    # PERCENTAGE_TO_REMOVE = 0.4
                    # num_to_change = int(np.ceil(PERCENTAGE_TO_REMOVE * num_ones)) # we REMOVE % of the edges
        
                    # # Randomly select indices to change
                    # np.random.seed(0) 
                    # indices_to_change = ones_indices[np.random.choice(num_ones, num_to_change, replace=False)]
        
                    # # Set the selected indices to 0 in the submatrix
                    # for index in indices_to_change: cluster[tuple(index)] = 0
        
                    # # Update the original matrix with the modified submatrix
                    
                    # for num_r,i in enumerate(cluster_indices):
                    #     for num_c,j in enumerate(cluster_indices):
                    #         adjacency_matrix[i,j] = cluster[num_r,num_c]# we update the adjacency matrix with the modified cluster


                    # Save adjacency_matrix to a CSV file
                    adjacency_matrix_df = pd.DataFrame(adjacency_matrix, index=indexes, columns=indexes) # we use the indexes of the samples to set the row and column names
                    
                    
                    csv_ending = str("hc_") + str(N_PCs) + "PCs_" + str(MIN_EDGES) + "-" + str(MAX_EDGES) + "_edges_bin"                    
                    
                    adjacency_matrix_df.to_csv("data/adj_matrices/adj_matrix_"+ csv_ending +".csv", index=True, sep=',')
                    # print("\nFinal Adjacency_matrix:\n", adjacency_matrix_df)

                    adjacency_matrix_df = pd.DataFrame(adjacency_matrix, index=indexes_with_clusters, columns=indexes_with_clusters)
                    # plot the graph
                    edge_index, edge_attr, sample_to_index_df = get_edges_from_adjacency(adjacency_matrix_df, print_tmp=False)
                    plot_gr(adj = adjacency_matrix_df, edges= edge_index, sample_to_index=sample_to_index_df, print_stats=False, csv_ending = csv_ending)
    if not BINARY_EDGES:
        for cutoff in 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60:
            
            print("Producing weighted edges Adjacency matrix with cutoff value: ", str(cutoff))

            adjacency_matrix = select_edges(sample_similarity_matrix, indexes_with_clusters, cluster_id, cutoff=cutoff, print_tmp=False, binary_edges=False)
            # print("\nFiltered similarity matrix:\n", adjacency_matrix)
                    
            row_sums = np.sum(adjacency_matrix, axis=1)
            # print("\nNumber of edges per each node after filtering:\n", row_sums)

            print("Sum of edges: ", np.sum(adjacency_matrix))
            print("Number of (undirected) edges > 0: ", int(np.sum(adjacency_matrix > 0)/2))

            adjacency_matrix_df = pd.DataFrame(adjacency_matrix, index=indexes, columns=indexes) # we use the indexes of the samples to set the row and column names

            path = "data/adj_matrices/"
            csv_ending = str("adj_matrix_hc_") + str(N_PCs) + "PCs_" + "rv_" + "cutoff_" + str(cutoff)
            full_path = path + csv_ending + ".csv"

            print("Saving the adjacency matrix to: ", full_path, "\n")
            adjacency_matrix_df.to_csv(full_path, index=True, sep=',')
            # print("\nFinal Adjacency_matrix:\n", adjacency_matrix_df)

            adjacency_matrix_df = pd.DataFrame(adjacency_matrix, index=indexes_with_clusters, columns=indexes_with_clusters)
            # plot the graph
            edge_index, edge_attr, sample_to_index_df = get_edges_from_adjacency(adjacency_matrix_df, print_tmp=False)
            plot_gr(adj = adjacency_matrix_df, edges= edge_index, sample_to_index=sample_to_index_df, print_stats=False, csv_ending = csv_ending)


    # find the highly connected node
    selected_row = adjacency_matrix[np.argmax(row_sums)] # adjacency_matrix.iloc[np.argmax(row_sums)]

if __name__ == '__main__':
    main()


