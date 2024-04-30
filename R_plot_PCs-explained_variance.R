

ev = read.table("Desktop/Biosust_CEH/data_Shyam/PCA_Moiz/principalComponents_ofFish_basedOnGWAS_EIGENVALUES.tsv")
sum(ev)
ev/sum(ev)

plot(ev$V1)
plot(cumsum(ev$V1)/sum(ev$V1))
abline(h=0.5)