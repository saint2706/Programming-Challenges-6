# Independent oracle: base-R test results on the bundled sample data, written as JSON.
# Run from the challenge folder:  Rscript scripts/make_reference.R
# Needs the jsonlite package (install.packages("jsonlite")). Output: tests/reference/r_tests.json

args <- commandArgs(trailingOnly = FALSE)
script_arg <- sub("^--file=", "", args[grep("^--file=", args)])
root <- normalizePath(file.path(dirname(script_arg), ".."), winslash = "/")
data_dir <- file.path(root, "sample_data")
out <- file.path(root, "tests", "reference", "r_tests.json")
dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)

library(jsonlite)

pick <- function(x) list(
  statistic = unname(as.numeric(x$statistic)),
  parameter = if (is.null(x$parameter)) NULL else unname(as.numeric(x$parameter)),
  p_value = as.numeric(x$p.value),
  conf_int = if (is.null(x$conf.int)) NULL else as.numeric(x$conf.int),
  estimate = if (is.null(x$estimate)) NULL else unname(as.numeric(x$estimate))
)

sleep <- read.csv(file.path(data_dir, "sleep.csv"))
iris <- read.csv(file.path(data_dir, "iris.csv"))
tooth <- read.csv(file.path(data_dir, "toothgrowth.csv"))
ucb <- read.csv(file.path(data_dir, "ucb_admissions.csv"))

res <- list()

# Welch and Student two-sample t on sleep (group 1 vs group 2)
g1 <- sleep$extra[sleep$group == 1]
g2 <- sleep$extra[sleep$group == 2]
res$welch_t_sleep <- pick(t.test(g1, g2, var.equal = FALSE))
res$student_t_sleep <- pick(t.test(g1, g2, var.equal = TRUE))
res$paired_t_sleep <- pick(t.test(g1, g2, paired = TRUE))

# Mann-Whitney on sleep (ties present, so normal approximation with continuity correction)
# R's default is an exact-style p-value even with ties (it differs from the normal approximation
# that scipy uses); the normal approximation with continuity correction is the like-for-like check.
res$mann_whitney_sleep <- pick(suppressWarnings(wilcox.test(g1, g2, conf.int = FALSE)))
res$mann_whitney_sleep_normal <- pick(suppressWarnings(
  wilcox.test(g1, g2, exact = FALSE, correct = TRUE, conf.int = FALSE)
))

# Welch ANOVA (oneway.test) and Kruskal-Wallis on iris petal length by species
res$welch_anova_iris_petal_length <- pick(oneway.test(
  petal_length ~ species, data = iris, var.equal = FALSE
))
res$anova_iris_petal_length <- pick(oneway.test(
  petal_length ~ species, data = iris, var.equal = TRUE
))
res$kruskal_wallis_iris_petal_length <- pick(kruskal.test(petal_length ~ species, data = iris))

# Chi-square without correction and Fisher's exact on fixed 2x2 tables
tab <- matrix(c(12, 5, 7, 16), nrow = 2)
res$chi2_2x2 <- pick(suppressWarnings(chisq.test(tab, correct = FALSE)))
res$fisher_2x2 <- pick(fisher.test(matrix(c(8, 2, 1, 5), nrow = 2), conf.int = TRUE))

# Mantel-Haenszel on UC Berkeley admissions, strata = department, no continuity correction
ucb_array <- xtabs(Freq ~ Admit + Gender + Dept, data = ucb)
res$cmh_ucb <- pick(mantelhaen.test(ucb_array, correct = FALSE))

# Correlations on ToothGrowth (dose vs length)
res$pearson_tooth <- pick(cor.test(tooth$dose, tooth$len, method = "pearson"))
res$spearman_tooth <- pick(suppressWarnings(
  cor.test(tooth$dose, tooth$len, method = "spearman", exact = FALSE)
))
res$kendall_tooth <- pick(suppressWarnings(
  cor.test(tooth$dose, tooth$len, method = "kendall", exact = FALSE)
))

# Multiplicity adjustments on a fixed vector
p <- c(0.001, 0.008, 0.039, 0.041, 0.2)
res$holm <- as.numeric(p.adjust(p, method = "holm"))
res$benjamini_hochberg <- as.numeric(p.adjust(p, method = "BH"))
res$multiplicity_input <- p

write_json(res, out, digits = 12, auto_unbox = TRUE, pretty = TRUE, null = "null")
cat("wrote", out, "\n")
