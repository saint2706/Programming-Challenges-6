# Reference numbers from R's survival package, used as the oracle for the Python tests.
#
#   Rscript scripts/make_reference.R        (needs R with survival and jsonlite)
#
# Writes tests/reference/r_survival.json. R is not needed to run the tests: the file is committed.
library(survival)
library(jsonlite)

root <- if (file.exists("sample_data/lung.csv")) "." else stop("run from the challenge folder")
lung <- read.csv(file.path(root, "sample_data/lung.csv"), na.strings = "NA")
gbsg <- read.csv(file.path(root, "sample_data/gbsg.csv"), na.strings = "NA")

km_block <- function(time, event, conf_type, entry = NULL, tau = NULL) {
  y <- if (is.null(entry)) Surv(time, event) else Surv(entry, time, event)
  fit <- survfit(y ~ 1, conf.type = conf_type)
  s <- summary(fit, censored = FALSE)
  q <- quantile(fit, c(0.25, 0.5, 0.75), conf.int = TRUE)
  out <- list(
    conf_type = conf_type,
    time = s$time, n_risk = s$n.risk, n_event = s$n.event,
    surv = s$surv, std_err = s$std.err, lower = s$lower, upper = s$upper,
    quantile = list(p = c(0.25, 0.5, 0.75), value = unname(q$quantile),
                    lower = unname(q$lower), upper = unname(q$upper))
  )
  if (!is.null(tau)) {
    tab <- summary(fit, rmean = tau)$table
    out$rmst <- list(tau = tau, mean = unname(tab["rmean"]), se = unname(tab["se(rmean)"]))
  }
  out
}

cases <- list()

# --- Kaplan-Meier: every interval type, overall and one subgroup, with and without late entry
cases$km <- list()
for (ct in c("log-log", "log", "plain", "arcsin")) {
  cases$km[[paste0("lung_all_", ct)]] <- km_block(lung$time, lung$event, ct, tau = 500)
}
male <- lung$sex == 1
cases$km$lung_male <- km_block(lung$time[male], lung$event[male], "log-log", tau = 500)
cases$km$lung_female <- km_block(lung$time[!male], lung$event[!male], "log-log", tau = 500)
cases$km$gbsg_all <- km_block(gbsg$rfstime, gbsg$event, "log-log", tau = 2000)

set.seed(2026)
entry <- lung$time * runif(nrow(lung)) * 0.8
cases$km$lung_late_entry <- c(
  list(entry = entry),
  km_block(lung$time, lung$event, "log-log", entry = entry)
)

# --- Log-rank family (survdiff): two groups, K groups, stratified, FH rho = 1
sd_block <- function(fit) {
  list(chisq = fit$chisq, df = length(fit$n) - 1, obs = as.vector(fit$obs),
       exp = as.vector(fit$exp), n = as.vector(fit$n))
}
cases$survdiff <- list(
  lung_sex = sd_block(survdiff(Surv(time, event) ~ sex, data = lung)),
  lung_sex_rho1 = sd_block(survdiff(Surv(time, event) ~ sex, data = lung, rho = 1)),
  gbsg_grade = sd_block(survdiff(Surv(rfstime, event) ~ grade, data = gbsg)),
  gbsg_grade_rho1 = sd_block(survdiff(Surv(rfstime, event) ~ grade, data = gbsg, rho = 1)),
  gbsg_hormon_strata_meno = sd_block(
    survdiff(Surv(rfstime, event) ~ hormon + strata(meno), data = gbsg)
  ),
  gbsg_grade_strata_hormon = sd_block(
    survdiff(Surv(rfstime, event) ~ grade + strata(hormon), data = gbsg)
  )
)

# --- Cox regression
cox_block <- function(fit, newdata = NULL, times = NULL) {
  s <- summary(fit)
  out <- list(
    names = names(coef(fit)), coef = unname(coef(fit)), se = unname(sqrt(diag(vcov(fit)))),
    loglik = fit$loglik[2], loglik_null = fit$loglik[1], score = fit$score,
    n = fit$n, nevent = fit$nevent, concordance = unname(s$concordance[1]),
    vcov = unname(vcov(fit))
  )
  out
}

lung_cc <- lung[complete.cases(lung[, c("age", "sex", "ph_ecog")]), ]
f_efron <- coxph(Surv(time, event) ~ age + sex + ph_ecog, data = lung_cc, ties = "efron")
f_breslow <- coxph(Surv(time, event) ~ age + sex + ph_ecog, data = lung_cc, ties = "breslow")
zph <- function(fit, tr) {
  z <- cox.zph(fit, transform = tr)
  list(transform = tr, names = rownames(z$table), chisq = unname(z$table[, "chisq"]))
}
cases$cox <- list(
  lung_efron = c(cox_block(f_efron), list(
    zph = lapply(c("km", "rank", "identity", "log"), function(tr) zph(f_efron, tr)))),
  lung_breslow = c(cox_block(f_breslow), list(zph = list(zph(f_breslow, "km"))))
)

form_g <- Surv(rfstime, event) ~ age + meno + size + grade + nodes + pgr + er + hormon
f_g <- coxph(form_g, data = gbsg)
cases$cox$gbsg_efron <- c(cox_block(f_g), list(
  zph = lapply(c("km", "rank"), function(tr) zph(f_g, tr))))

f_gs <- coxph(
  Surv(rfstime, event) ~ age + size + grade + nodes + pgr + er + strata(hormon), data = gbsg
)
cases$cox$gbsg_strata_hormon <- c(cox_block(f_gs), list(zph = list(zph(f_gs, "km"))))

lung_cc$entry <- entry[complete.cases(lung[, c("age", "sex", "ph_ecog")])]
f_lt <- coxph(Surv(entry, time, event) ~ age + sex + ph_ecog, data = lung_cc)
cases$cox$lung_late_entry <- c(cox_block(f_lt), list(zph = list(zph(f_lt, "km"))))

# baseline survival curves for two covariate profiles (Efron model, R's default Efron baseline)
nd <- data.frame(age = c(50, 70), sex = c(1, 2), ph_ecog = c(0, 2))
sf <- survfit(f_efron, newdata = nd)
sfs <- summary(sf, censored = FALSE)
cases$cox$lung_efron$predict <- list(
  newdata = nd, time = sfs$time, surv = unname(sfs$surv)
)
sf_b <- survfit(f_breslow, newdata = nd)
sfb <- summary(sf_b, censored = FALSE)
cases$cox$lung_breslow$predict <- list(newdata = nd, time = sfb$time, surv = unname(sfb$surv))

dir.create(file.path(root, "tests/reference"), recursive = TRUE, showWarnings = FALSE)
write_json(
  list(r_version = R.version.string, survival_version = as.character(packageVersion("survival")),
       cases = cases),
  file.path(root, "tests/reference/r_survival.json"),
  auto_unbox = TRUE, digits = NA, na = "null", pretty = TRUE
)
cat("wrote tests/reference/r_survival.json\n")
