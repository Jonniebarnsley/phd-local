install.packages("randtoolbox")
library(randtoolbox)

sample = sobol(n=1024, dim=6, start=0)
start <- 256
end <- start + 16
subsample <- sample[start:end,]
#plot(subsample)

pairs(subsample)

