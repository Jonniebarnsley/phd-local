#install.packages("randtoolbox")
library(randtoolbox)

ndim <- 6 # number of dimensions for sobol sample
# generate large number of samples and then subsample to the size you want
sample = sobol(n=1024, dim=ndim, start=1, init=TRUE)
start <- 16 # skip first 16 samples
numpoints <- 128
end <- start + numpoints
subsample <- sample[start:(end-1),]
pairs(subsample, cex=0.2)

# Transform function
transform_sobol <- function(sobol_points, lower, upper, log_scale = integer(0), recip_scale = integer(0)) {
  transformed <- sobol_points
  for (i in 1:ndim) {
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

lower <- c(1e9, 2, 1, 50, 9618.882299, 6e17) # lower bounds
upper <- c(1e17, 4, 5, 300, 159188.5414, 1e21) # upper bounds
param_names = c("aPhi", "n", "1/m", "uf", "gamma", "UMV")
log_scale <- c(1, 5, 6) # dimension numbers to sample on log scale

transformed <- transform_sobol(subsample, lower=lower, upper=upper, log_scale=log_scale)
transformed

pairs(transformed[1:16,], log=log_scale, labels=param_names)
pairs(
  transformed, 
  cex=0.2, 
  log=log_scale, 
  labels=param_names
  )

df <- as.data.frame(transformed)
colnames(df) <- param_names
df['A0_multiplier'] <- 1e5**(3-df['n'])
df['m'] <- 1/df['1/m'] # convert into 1/m format for BISICLES

model_names <- c("CESM2-WACCM", "MRI-ESM2-0", "MIROC-ES2L", "GISS-E2-1-G",
                  "ACCESS-ESM1-5", "CanESM5", "EC-Earth3-Veg", "IPSL-CM6A-LR")

info <- data.frame(
  name = sprintf("run%03d", 1:128),
  model = rep(model_names, each = 16)
)

final <- cbind(info, df)
final

# Save to CSV
setwd("~/code/phd/local")
write.csv(final, "data/sobol_samples.csv", row.names = FALSE, quote = FALSE)

