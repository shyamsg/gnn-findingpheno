

ev = read.table("data/PCA/principalComponents_ofFish_basedOnGWAS_EIGENVALUES.tsv")
sum(ev)
ev/sum(ev)

explained_variance = cumsum(ev$V1)/sum(ev$V1)

plot(cumsum(ev$V1)/sum(ev$V1), xlab = "PC number", ylab = "Variance explained")
abline(h=0.5)
# plot()

explained_variance = cumsum(ev$V1)/sum(ev$V1)

# Find indexes for the first value greater than each threshold
index_0_25 <- which(explained_variance > 0.25)[1]
index_0_5 <- which(explained_variance > 0.5)[1]
index_0_75 <- which(explained_variance > 0.75)[1]
cat("Index for value > 0.25:", index_0_25, "\n") # 8
cat("Index for value > 0.5:", index_0_5, "\n") # 48
cat("Index for value > 0.75:", index_0_75, "\n") # 132
