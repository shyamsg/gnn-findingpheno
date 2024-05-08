import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix


def get_edges_from_adjacency(adjacency_matrix, print_tmp=False):
    """
    Get the similarity graph from the adjacency_matrix.
    
    Parameters:
    - adjacency_matrix (numpy.ndarray): matrix n_samples x n_samples whose values represent similarities.
    - print_tmp (bool): Option to print intermediate results
    
    Returns:
    - edge_index (torch.Tensor): Tensor of shape [2, num_edges] where num_edges is the total number of edges in the graph. It consists of two rows. The first row contains the indices of the source nodes of each edge. The second row contains the indices of the destination nodes of each edge.
    - edge_attr (torch.Tensor): Weights of the edges [num_edges]
    - sample_to_index_df (pd.DataFrame): # Dataframe with the sample names (ex: F002) in the column "sample" and the node ordering (0 - n_samples-1) in the column "graph_index"

    """

    indexes = adjacency_matrix.index

    sample_to_index = {sample: i for i, sample in enumerate(indexes)} 
    if print_tmp: print("\nSample to index:\n", sample_to_index)
    sample_to_index_df = pd.DataFrame(sample_to_index.items(), columns=['sample', 'graph_index'])
    

    sparse_adj_matrix = sp.csr_matrix(adjacency_matrix)
    if print_tmp: print("\nSHAPE of sparse_adj_matrix\n", sparse_adj_matrix.shape)
    if print_tmp: print("\nsparse_adj_matrix\n", sparse_adj_matrix)
    
    # Save sparse_adj_matrix to a CSV file
    # df = pd.DataFrame({'tuples': [str(t) for t in sparse_adj_matrix.indices], 'values': sparse_adj_matrix.data})
    # df.to_csv("/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/output/sparse_adj_matrix.csv", index=True)

    edge_index, edge_attr = from_scipy_sparse_matrix(sparse_adj_matrix)

    return edge_index, edge_attr, sample_to_index_df


def plot_gr(adj=None, edges= None, sample_to_index=None, print_stats=False, csv_ending=None):
    
    """
    Plot a graph using real sample names as labels
    
    Parameters:
    - adj: matrix n_samples x n_samples whose values represent similarities.
    - edges: matrix or dataframe representing edges with columns=['source', 'target']
    - sample_to_index: matrix or dataframe with sample names as keys and sample indexes as values
    - print_stats (bool): Option to print
    
    Returns:
    """

    # All the following files are in the google colab folder "FindingPheno/output"
    
    # adj = '/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/output/adjacency_matrix.csv'
    # edges = '/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/output/edge_index.csv'
    # sample_to_index = '/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/output/sample_to_index.csv'
    
    if adj is None and edges is None:
        raise ValueError("At least one of the following must be provided: adj, edges")


    if adj is not None: 
        if isinstance(adj, str): adj = pd.read_csv(adj, header=0, index_col=0)
    if edges is not None: 
        if isinstance(edges, str): edges = pd.read_csv(edges, index_col=0)
    if sample_to_index is not None:
        if isinstance(sample_to_index, str): sample_to_index =pd.read_csv(sample_to_index)

    if (edges is None or sample_to_index is None) and adj is not None: 
        edge_index, edge_attr, sample_to_index_df = get_edges_from_adjacency(adj, print_tmp=False)

    if not isinstance(edges, pd.DataFrame):
        edges = pd.DataFrame(edges.numpy().T, columns=['source', 'target'])

    if not isinstance(sample_to_index, pd.DataFrame):
        sample_to_index = pd.DataFrame(sample_to_index.items(), columns=['sample', 'graph_index'])

    

    # Create a dictionary with the sample ordering (0-n_samples) as keys and the sample codes (ex: F002) as values
    sample_names = dict(zip(sample_to_index.iloc[:, 1], sample_to_index.iloc[:, 0]))

    # Translate all edges using the dictionary "sample_names"
    edges['source'] = edges['source'].map(sample_names)
    edges['target'] = edges['target'].map(sample_names)

    graph = nx.Graph()

    if adj is not None: # add nodes from the adjacency matrix
        graph.add_nodes_from(adj.columns.tolist())
    else: # add nodes from the edges list
        # we can create the node list from the edges, since each node has at least one edge
        graph.add_nodes_from(edges['source'])
        graph.add_nodes_from(edges['target']) # this shouldn't be necessary (graph is undirected), just to be sure
    if print_stats: print("graph nodes: \n", graph.nodes())
        
    # graph.add_edges_from(edges[['source', 'target']].values)
    for edge in edges.values:
        graph.add_edge(edge[0], edge[1])
    if print_stats: print("graph edges: \n", graph.edges()) 

    if print_stats: print("number of nodes:", graph.number_of_nodes())
    if print_stats: print("number of edges:", graph.number_of_edges())


    # Draw the graph with node labels
    plt.figure(figsize=(10, 8))
    nx.draw(graph, with_labels=True, node_size=20, node_color='skyblue', edge_color="black", font_size=5, font_color='red')#, labels= sample_names)
    plt.title('Graph of Samples with Real Sample Names')
    if csv_ending is not None: plt.text(0.5, 1.05, csv_ending, horizontalalignment='center', fontsize=12, transform=plt.gca().transAxes)
    # plt.show()
    
    if csv_ending is not None:
        plt.savefig('/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/graph_plots/graph_'+csv_ending+'.png')
    else:
        plt.savefig('/Users/lorenzoguerci/Desktop/Biosust_CEH/FindingPheno/data/graph_plots/graph.png')


def plot_feature_correlation(data):
    """
    Plot the correlation matrix of features in the given data.

    Parameters:
    - data (pd.DataFrame): Dataframe containing the features.

    Returns:
    - None
    """
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title('Correlation Matrix of Features')
    plt.xticks(range(len(data.columns)), data.columns, rotation=90)
    plt.yticks(range(len(data.columns)), data.columns)
    plt.show()