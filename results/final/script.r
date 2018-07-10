setwd("/home/mad_otter/workspace/Intermediate-model-SOM-FPNA/results/final")
setwd("/users/yabernar/workspace/results/final")
a <- read.table("3x3_9n", header=TRUE, sep=";", dec=".")
b <- read.table("6x6_12n", header=TRUE, sep=";", dec=".")
c <- read.table("9x9_9n", header=TRUE, sep=";", dec=".")
d <- read.table("3x3_9n_dyn", header=TRUE, sep=";", dec=".")
e <- read.table("9x9_9n_dyn", header=TRUE, sep=";", dec=".")

a_s = split(a, a$image)
b_s = split(b, b$image)
c_s = split(c, c$image)
d_s = split(d, d$image)
e_s = split(e, e$image)
i = 3
axis_scale = 1.5
main_scale = 2
label_scale = 2

c1 <- rainbow(3)
c2 <- rainbow(3, alpha=0.2)
c3 <- rainbow(3, v=0.7)

par(mfrow=c(3,4), mar=c(1.5,1,1.5,1), oma = c(1,2,0,0) ,xpd = NA)

boxplot(a_s[[i]]$mean_error~a_s[[i]]$connexion, main="mean\ error", col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')
title(ylab=expression(bold("3x3")), line=0, cex.lab=2, family="Calibri Light")
boxplot(a_s[[i]]$psnr~a_s[[i]]$connexion, main="psnr", col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')
boxplot(a_s[[i]]$differential~a_s[[i]]$connexion, main="differential comp.", col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')
boxplot(a_s[[i]]$compression~a_s[[i]]$connexion, main="total\ compression", col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')

boxplot(b_s[[i]]$mean_error~b_s[[i]]$connexion, col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')
title(ylab=expression(bold("6x6")), line=0, cex.lab=2, family="Calibri Light")
boxplot(b_s[[i]]$psnr~b_s[[i]]$connexion, col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')
boxplot(b_s[[i]]$differential~b_s[[i]]$connexion, col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')
boxplot(b_s[[i]]$compression~b_s[[i]]$connexion, col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')

boxplot(c_s[[i]]$mean_error~c_s[[i]]$connexion, col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')
title(ylab=expression(bold("9x9")), line=0, cex.lab=2, family="Calibri Light")
boxplot(c_s[[i]]$psnr~c_s[[i]]$connexion, col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')
boxplot(c_s[[i]]$differential~c_s[[i]]$connexion, col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')
boxplot(c_s[[i]]$compression~c_s[[i]]$connexion, col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')

# kruskal wallis p-values

i = 2
t_s = b_s
kruskal.test(t_s[[i]]$mean_error ~ t_s[[i]]$connexion)
kruskal.test(t_s[[i]]$psnr ~ t_s[[i]]$connexion)
kruskal.test(t_s[[i]]$differential ~ t_s[[i]]$connexion)
kruskal.test(t_s[[i]]$compression ~ t_s[[i]]$connexion)



kruskal.test(t_s[[i]]$mean_error~t_s[[i]]$connexion)
kruskal.test(t_s[[i]]$psnr~t_s[[i]]$connexion)
kruskal.test(t_s[[i]]$~t_s[[i]]$connexion)


# histograms

histo2 <- data.frame(t(histo[-1]))
colnames(histo2) <- histo[, 1]
lake = histo2[1:3]
pepper = histo2[4:6]
plot(lake)
lake.sorted <- apply(lake,2,sort,decreasing=TRUE)
pepper = sort.default(pepper, decreasing = TRUE)

# Grouped barplot
library(ggplot2)
ggplot(lake.sorted, aes(x = "identity", y = Length)) + geom_bar(aes(fill = Amount, order = Location), stat = "identity") 

t2 = t(lake.sorted)
t3 = matrix(t2)
barplot(t2[3, 1:81], col=rainbow(3), border="white", font.axis=2, beside=T, legend=colnames(t3), xlab="group", font.lab=2)

barplot(lake.sorted)
plot(lake.sorted[1:81, 2])
plot(lake.sorted[1:81, 3])

par(mfrow=c(1,1))
lines(lake, type='l') 

# dynamic connexions boxplot
par(mfrow=c(2,2), mar=c(1.5,1,1.5,1), oma = c(1,2,0,0) ,xpd = NA)

c1 <- rainbow(5)
c2 <- rainbow(5, alpha=0.2)
c3 <- rainbow(5, v=0.7)

i = 5

boxplot(d_s[[i]]$mean_error~d_s[[i]]$connexion, main="mean\ error", col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE)
# title(ylab=expression(bold("3x3")), line=0, cex.lab=2, family="Calibri Light")
boxplot(d_s[[i]]$psnr~d_s[[i]]$connexion, main="psnr", col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')
boxplot(d_s[[i]]$differential~d_s[[i]]$connexion, main="differential comp.", col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')
boxplot(d_s[[i]]$compression~d_s[[i]]$connexion, main="total\ compression", col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')

boxplot(e_s[[i]]$mean_error~e_s[[i]]$connexion, main="mean\ error", col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE)
title(ylab=expression(bold("9x9")), line=0, cex.lab=2, family="Calibri Light")
boxplot(e_s[[i]]$psnr~e_s[[i]]$connexion, main="psnr", col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')
boxplot(e_s[[i]]$differential~e_s[[i]]$connexion, main="differential comp.", col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')
boxplot(e_s[[i]]$compression~e_s[[i]]$connexion, main="total\ compression", col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2, cex.axis=axis_scale, cex.main=main_scale, cex.lab =label_scale, horizontal = TRUE, yaxt='n')
