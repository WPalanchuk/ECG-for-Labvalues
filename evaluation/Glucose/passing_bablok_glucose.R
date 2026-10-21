library(mcr)

data <- read.csv("passing_bablok_glucose_nonnegative.csv")

pb <- mcreg(
  data$reference,
  data$prediction,
  method.reg = "PaBa",
  method.ci = "bootstrap",
  method.bootstrap.ci = "quantile",
  nsamples = 1000,
  rng.seed = 42,
  mref.name = "Reference",
  mtest.name = "Prediction"
)

print(getCoefficients(pb))

pdf("passing_bablok_glucose.pdf", width = 6, height = 6)

plot(
  pb,
  main = "",
  add.cor = FALSE
)

dev.off()