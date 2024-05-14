import pandas as pd
from scipy.spatial.distance import pdist, squareform

def find_closest_distances(file_path, num_closest=8, output_file_path='gene_closest_distances.csv'):
    # Read data from CSV file
    data = pd.read_csv(file_path)

    # Extract sample IDs and gene data
    sample_id = data['sample_id']
    genes_data = data.drop(columns=['sample_id'])

    # Calculate pairwise Euclidean distances between genes
    gene_distances = pdist(genes_data, metric='euclidean')
    gene_distance_matrix = squareform(gene_distances)

    results = []

    # Iterate over genes to find closest distances
    for i, gene1 in enumerate(genes_data.columns):
        sorted_distances = sorted([(gene_distance_matrix[i, j], gene2) for j, gene2 in enumerate(genes_data.columns) if i != j])
        closest_distances = sorted_distances[:num_closest]

        # Append results to list
        for distance, gene2 in closest_distances:
            results.append([gene1, gene2, distance])

    # Create DataFrame from results
    result_df = pd.DataFrame(results, columns=['Gene1', 'Gene2', 'Distance'])

    # Save results to CSV file
    result_df.to_csv(output_file_path, index=False)

    print("Closest gene distances saved to:", output_file_path)

if __name__ == "__main__":
    # Specify file path for input data
    file_path = 'scaled_input.csv'

    # Call the function to find closest distances
    find_closest_distances(file_path)
