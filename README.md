# gnn-findingpheno

Plan: https://docs.google.com/document/d/1VBOz0LpTis6ROQ3tGJn4juo0FtcpoCyavlUohhSdbcY/edit

## Example colab for a course
Colab away [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a0Po-kH1aZlZ6BC9d3TnPKVzeMnkuKYh)


Pipeline:

(1a) clustering.py:  Filter samples to use in the analysis:<br>
*INPUT: data/PCA/principalComponents_ofFish_basedOnGWAS.xlsx
*OUTPUT: data/PCA/PCs_Fish_GWAS-based_cluster-filtered.csv
(1b) T_features_select.py:   Select Trascriptome features
    INPUT: data/HoloFish_HostRNA_normalised_GeneCounts_230117.csv
    OUTPUT: data/T_features/T_selected_features_" + SELECTION_METHOD + ".csv"

(2) get_Adjacency_matrix_from_GWAS-based_PCs.py: Get the ADJACENCT MATRIX to obtain the GCN-graph 
    INPUT: data/PCA/PCs_Fish_GWAS-based_cluster-filtered.csv
    OUTPUT: data/adj_matrices/" + str("adj_matrix_hc_") + str(N_PCs) + "PCs_" + "rv_" + str(cutoff)

(3)GCN_nodePred_similarityMatrix.py
    INPUT: data/adj_matrices/SPECIFIC_ADJ_FILE, data/HoloFish_MetaG_MeanCoverage_20221114.csv (MG), T_selected_features_T_SELECTION_ALGORTIHM (T), data/HoloFish_FishVariables_20221116.csv (Pheno)
    OUTPUT: predictions, loss