library(mcr)

data1 <- read.csv("passing_bablok_1lead_nonnegative.csv")
data12 <- read.csv("passing_bablok_12lead_nonnegative.csv")

cat("Starting 1-lead Passing-Bablok...\n")

pb1 <- mcreg(
  data1$reference,
  data1$prediction,
  method.reg = "PaBa",
  method.ci = "bootstrap",
  method.bootstrap.ci = "quantile",
  nsamples = 1000,
  rng.seed = 42,
  mref.name = "Reference",
  mtest.name = "Prediction"
)

print(getCoefficients(pb1))

pdf("passing_bablok_1lead.pdf", width = 6, height = 6)

plot(
  pb1,
  main = "",
  add.cor = FALSE
)

dev.off()

cat("1-lead figure finished: passing_bablok_1lead.pdf\n")

cat("Starting 12-lead Passing-Bablok...\n")

pb12 <- mcreg(
  data12$reference,
  data12$prediction,
  method.reg = "PaBa",
  method.ci = "bootstrap",
  method.bootstrap.ci = "quantile",
  nsamples = 1000,
  rng.seed = 42,
  mref.name = "Reference",
  mtest.name = "Prediction"
)

print(getCoefficients(pb12))

pdf("passing_bablok_12lead.pdf", width = 6, height = 6)

plot(
  pb12,
  main = "",
  add.cor = FALSE
)

dev.off()

cat("12-lead figure finished: passing_bablok_12lead.pdf\n")