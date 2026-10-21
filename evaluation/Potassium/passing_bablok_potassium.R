library(mcr)

data <- read.csv("passing_bablok_potassium_nonnegative.csv")

cat("Starting potassium Passing-Bablok...\n")

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

pdf("passing_bablok_potassium.pdf", width = 6, height = 6)

plot(
  pb,
  main = "",
  add.cor = FALSE
)

dev.off()

cat("Figure finished: passing_bablok_potassium.pdf\n")