from torch_geometric.data import Data
import numpy as np
import pandas as pd
from scipy.spatial import distance
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
import matplotlib.pyplot as plt

from similarity_graph_utilities import get_edges_from_adjacency
from similarity_graph_utilities import plot_gr


def get_similarity_matrix(sample_pcs, n_samples, print_tmp=False):
    """
    Get n_sample x n_sample similarity matrix from n_sample x n_PCs matrix
    Eucledian distance: based on normal distribution. Natural connection between Gaussian kernels and Euclinead distance

    Parameters:
    - parameter1 (type): Description of parameter1
    
    Returns:
    - return_type: Sample_similarity_matrix
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
    sample_similarity_matrix = np.ones((n_samples,n_samples))-dist_matrix
    # We want 0s insted of 1s for the diagonal
    np.fill_diagonal(sample_similarity_matrix, 0)
    if print_tmp: print("\nMatrix after 1-dist_matrix and 0s in the diagonal operation\n",sample_similarity_matrix)

    df = pd.DataFrame(sample_similarity_matrix)
    df.to_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/PCA_Moiz/sample_similarity_matrix.tsv", sep='\t', index=True, header=True)

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


def select_edges(sample_similarity_matrix, cutoff, min_edges=12, max_edges=25, print_tmp=False):
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
    for i, row_sum in enumerate(row_sums):
       print("Sum of row {}: {}".format(i+1, row_sum)) # OUTPUT should be >= min_edges for each line


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
    
    print(np.sum(filtered_similarity_matrix))

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




def main():

    N_PCs = 50 # Defined after using the R script (see google colab)

    sample_pcs = pd.read_excel("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/PCA_Moiz/principalComponents_ofFish_basedOnGWAS.xlsx", header=0, index_col=0)
    print(sample_pcs)
    #sample_pcs = pd.read_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/PCA_Moiz/principalComponents_ofFish_basedOnGWAS.xlsx", sep='\t', header=0)

    ### We want to keep only the samples for which we have transcriptomics and metagenomics data. 'final_input' is the file that contains all transcriptomics and metagenomics data for the samples we have such data for.
    MG_T_Ph = pd.read_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/T_MG_P_input_data/final_input.csv", header=0, index_col=0)
    # get the IDs of the sample with metagenomics and transcriptomics data
    samples_with_MG_T_Ph_data = list(MG_T_Ph.index)
    # print(samples_with_MG_T_Ph_data)

    # print(sample_pcs.index)
    # print(len(sample_pcs.index))

    # FILTERING for MG T data availability : keep only the samples (rows of sample_pcs) whose index is in samples_with_MG_T_Ph_data (samples with metagenomics and transcriptomics data)
    sample_pcs = sample_pcs[sample_pcs.index.isin(samples_with_MG_T_Ph_data)]

    # print("\n\n\n", sample_pcs.index)
    # print(len(sample_pcs.index))

    N_SAMPLES= sample_pcs.shape[0]
    # print(N_SAMPLES) # 207 after filtering, 361 before filtering

    indexes = sample_pcs.index


    # We select the first N_PCs features, N_PCs = 50 account for ~50% variability (see R file for the PC analysis)
    sample_pcs = sample_pcs.iloc[:,1:1 + N_PCs]
    #sample_pcs = sample_pcs.iloc[0:4,0:3] # for DEBUG


    sample_similarity_matrix = get_similarity_matrix(sample_pcs, n_samples=N_SAMPLES, print_tmp=False)
    print("\nSample_similarity_matrix:\n", sample_similarity_matrix)
    # print("SHAPE:", sample_similarity_matrix.shape) # (361,361)
    # print("SHAPE:",sample_similarity_matrix.flatten() > 0.4)

    rows, cols = sample_similarity_matrix.shape
    # Extract the upper diagonal elements into a vector
    upper_diag_vector = sample_similarity_matrix[np.triu_indices(rows, k=1)]
    # analyze_similarity(upper_diag_vector) # UNCOMMENT to plot a histogram of similarity-based edges distribution in order to choose a cutoff value
    CUTOFF = 0.60 # we choose a cutoff of 0.6
    # print("cutoff matrix:\n",sum(sample_similarity_matrix > CUTOFF)) # shape will be (n_samples,)
    
    adjacency_matrix = select_edges(sample_similarity_matrix, cutoff=CUTOFF, min_edges=12, max_edges=25, print_tmp=False)
    print("\nFiltered similarity matrix:\n", adjacency_matrix)
    
    row_sums = np.sum(adjacency_matrix, axis=1)
    print("\nNumber of edges per each node after filtering:\n", row_sums)

    # Save adjacency_matrix to a CSV file
    adjacency_matrix_df = pd.DataFrame(adjacency_matrix, index=indexes, columns=indexes) # we use the indexes of the samples to set the row and column names
    adjacency_matrix_df.to_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/output/adjacency_matrix_min12_max25.csv", index=True, sep=',')
    print("\nFinal Adjacency_matrix:\n", adjacency_matrix_df)

    # plot the graph
    edge_index, edge_attr, sample_to_index_df = get_edges_from_adjacency(adjacency_matrix_df, print_tmp=False)
    plot_gr(adj = adjacency_matrix_df, edges= edge_index, sample_to_index=sample_to_index_df, print_stats=False)

if __name__ == '__main__':
    main()


