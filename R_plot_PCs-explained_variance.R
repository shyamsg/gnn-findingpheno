

ev = read.table("data/PCA_Moiz/principalComponents_ofFish_basedOnGWAS_EIGENVALUES.tsv")
sum(ev)
ev/sum(ev)

#plot(ev$V1)

plot(cumsum(ev$V1)/sum(ev$V1), xlab = "PC number", ylab = "Variance explained")
abline(h=0.5)
plot()
