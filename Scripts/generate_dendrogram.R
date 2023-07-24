library(ggplot2)
library(ggdendro)
library(factoextra)

arguments <- commandArgs(trailingOnly = TRUE)
  # Read it in by using the arguments
  inputDirectory <- arguments[1]

df <- read.csv(file.path(inputDirectory, "Clustering", "clustering.csv"), header=TRUE, row.names ='workload')
hc <- hclust(dist(df, method="manhattan"), method="average")
pdf(file = file.path(inputDirectory, "Clustering", "silhouette.pdf"),width=4, height=3 )
par(mar = c(0.1, 0.1, 0.1, 0.1))
fviz_nbclust(df, FUN = hcut, method = "silhouette", main=NULL) + 
  labs(title="")
dev.off()

pdf(file = file.path(inputDirectory, "Clustering", "clustering.pdf"),width=9, height=8)
par(mar = c(0.1, 4, 0.1, 0.1))
plot(hc, cex=0.6, main=NULL, xlab="", sub="", ylab="Distance")
rect.hclust(hc, k = 2, border = 2:5)
dev.off()
