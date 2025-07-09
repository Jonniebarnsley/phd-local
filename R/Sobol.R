install.packages("randtoolbox")
library(randtoolbox)

sample = sobol(n=1024, dim=6, start=1, init=TRUE)
start <- 16
numpoints <- 512
end <- start + numpoints
subsample <- sample[start:(end-1),]
subsample
pairs(subsample, cex=0.2, main="CanESM5")

plot(subsample, xlim=c(0, 1), ylim=c(0, 1), main=paste("N =", numpoints))
abline(v = seq(0, 1, length.out = 9), col = "gray", lty = 2)

plot(subsample, xlim=c(0, 1), ylim=c(0, 1), xaxs = "i", yaxs = "i", main=paste("N =", numpoints))
abline(h = seq(0, 1, length.out = 9), col = "gray", lty = 2)

plot(subsample, xlim=c(0, 1), ylim=c(0, 1), xaxs = "i", yaxs = "i", main=paste("N =", numpoints))
abline(v = seq(0, 1, length.out = numpoints/4+1), col = "gray", lty = 2)
abline(h = seq(0, 1, length.out = numpoints/4+1), col = "gray", lty = 2)

pairs(subsample, cex=0.5)

# Transform function
transform_sobol <- function(sobol_points, lower, upper, log_scale = integer(0), recip_scale = integer(0)) {
  transformed <- sobol_points
  for (i in 1:6) {
    if (i %in% log_scale) {
      # Log-scale transformation
      transformed[, i] <- exp(log(lower[i]) + sobol_points[, i] * (log(upper[i]) - log(lower[i])))
    } else {
      # Linear transformation
      transformed[, i] <- lower[i] + sobol_points[, i] * (upper[i] - lower[i])
    }
  }
  return(transformed)
}

lower <- c(1e-2, 2, 1, 9618.882299, 50, 6e17)
upper <- c(1e6, 4, 5, 159188.5414, 300, 1e21)

transformed <- transform_sobol(subsample, lower=lower, upper=upper, log_scale=c(1, 3, 6))
transformed
pairs(transformed[1:16,], log=c(1, 3, 6))
pairs(transformed, cex=0.5, log=c(1, 3, 6))

param_names = c("aC", "n", "m", "gamma", "u0", "UMV")
df <- as.data.frame(transformed)
colnames(df) <- param_names

# Save to CSV
setwd("~/code/phd/local")
write.csv(df, "data/sobol_samples.csv", row.names = FALSE)

