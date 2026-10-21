library(mcr)

data_minmax <- read.csv("passing_bablok_minmax_nonnegative.csv")
data_standard <- read.csv("passing_bablok_standard_nonnegative.csv")

cat("Starting MinMax Passing-Bablok...\n")

pb_minmax <- mcreg(
  data_minmax$reference,
  data_minmax$prediction,
  method.reg = "PaBa",
  method.ci = "bootstrap",
  method.bootstrap.ci = "quantile",
  nsamples = 1000,
  rng.seed = 42,
  mref.name = "Reference",
  mtest.name = "Prediction"
)

print(getCoefficients(pb_minmax))

pdf("passing_bablok_minmax.pdf", width = 6, height = 6)

plot(
  pb_minmax,
  main = "",
  add.cor = FALSE
)

dev.off()

cat("MinMax figure finished: passing_bablok_minmax.pdf\n")

cat("Starting Standard Passing-Bablok...\n")

pb_standard <- mcreg(
  data_standard$reference,
  data_standard$prediction,
  method.reg = "PaBa",
  method.ci = "bootstrap",
  method.bootstrap.ci = "quantile",
  nsamples = 1000,
  rng.seed = 42,
  mref.name = "Reference",
  mtest.name = "Prediction"
)

print(getCoefficients(pb_standard))

pdf("passing_bablok_standard.pdf", width = 6, height = 6)

plot(
  pb_standard,
  main = "",
  add.cor = FALSE
)

dev.off()

cat("Standard figure finished: passing_bablok_standard.pdf\n")