# Writes three historical tables from R's datasets package to the folder given as the argument.
# Called by scripts/fetch_data.py: Rscript export_r_datasets.R <out-dir>
args <- commandArgs(trailingOnly = TRUE)
out <- args[1]
write.csv(sleep, file.path(out, "sleep.csv"), row.names = FALSE)
write.csv(ToothGrowth, file.path(out, "toothgrowth.csv"), row.names = FALSE)
write.csv(as.data.frame(UCBAdmissions), file.path(out, "ucb_admissions.csv"), row.names = FALSE)
