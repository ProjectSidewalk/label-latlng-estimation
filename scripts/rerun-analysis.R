# Re-run of the 2021 label lat/lng estimation analysis on the reconstructed dataset.
#
# This is scripts/label-latlng-estimation.Rmd distilled into a plain script: same cleaning
# filters, same seed and train/test split, same seven candidate estimators, same error metrics.
# The Rmd itself is left untouched as the frozen 2021 record; this script exists to produce a
# machine-readable baseline (tests/fixtures/r-baseline/) that the Python port is tested against.
# See https://github.com/ProjectSidewalk/label-latlng-estimation/issues/1 and /issues/2.
#
# Deviations from the Rmd, all deliberate:
#   1. Reads the reconstructed data/labels-*-latlng.csv.gz (issue #1) instead of the lost 2021
#      CSVs, and filters time_created < 2021-01-01 UTC: the original extraction was dated
#      2021-01-01, and rows stamped computation_method='depth' with later creation times exist
#      in production (evolution 93 stamped everything that had lat/lng when it ran) — those
#      cannot have been in the 2021 dataset and may not be genuine depth estimates. Rows with
#      NULL time_created are KEPT: they are early DC-deployment rows that predate the
#      time_created column (82,791 rows), all far older than the cutoff.
#   2. Vectorized geosphere calls instead of the Rmd's rowwise()/multidplyr clusters — identical
#      math, minutes faster.
#   3. zoom is read as integer and converted to a factor with fixed levels 1/2/3 (the Rmd's
#      per-file col_factor() level inference is order-of-appearance and non-deterministic across
#      files; the fit is identical, only the dummy-coding baseline could differ).
#   4. On top of the Rmd's train-set fits, est7 is also fit on the full cleaned data
#      (fixture est7_full) to give the Python equivalence tests a split-independent target.
#
# Usage: Rscript scripts/rerun-analysis.R   (from the repo root or scripts/)

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(tibble); library(purrr)
  library(geosphere); library(jsonlite)
})

# Resolve repo root whether run from root or scripts/.
root <- if (dir.exists("data")) "." else ".."
data_dir <- file.path(root, "data")
fix_dir  <- file.path(root, "tests", "fixtures", "r-baseline")
dir.create(fix_dir, recursive = TRUE, showWarnings = FALSE)

MAX_LABELS_PER_PANO <- 20
MAX_DIST_FROM_PANO  <- 50
TRAINING_FRAC       <- 0.8
CUTOFF <- as.POSIXct("2021-01-01 00:00:00", tz = "UTC")

col_types <-
  cols(label_id = col_integer(),
       label_type = col_factor(),
       lat = col_double(),
       lng = col_double(),
       panorama_lat = col_double(),
       panorama_lng = col_double(),
       canvas_x = col_integer(),
       canvas_y = col_integer(),
       canvas_width = col_integer(),
       canvas_height = col_integer(),
       heading = col_double(),
       pitch = col_double(),
       zoom = col_integer(),           # Rmd: col_factor(); see deviation 3
       photographer_heading = col_double(),
       photographer_pitch = col_double(),
       sv_image_x = col_integer(),
       sv_image_y = col_integer(),
       gsv_panorama_id = col_character(),
       street_edge_id = col_integer(),
       deleted = col_logical(),
       tutorial = col_logical(),
       computation_method = col_character(),
       pano_width = col_integer(),
       pano_height = col_integer(),
       time_created = col_datetime(),
       current_pano_x = col_integer(),
       current_pano_y = col_integer())

read_city <- function(city) {
  read_csv(file.path(data_dir, sprintf("labels-%s-latlng.csv.gz", city)),
           col_types = col_types) %>% mutate(city = city)
}

# Same city order as the Rmd (dc, seattle, newberg, columbus, spgg, cdmx, pittsburgh).
cities <- c("dc", "seattle", "newberg", "columbus", "spgg", "cdmx", "pittsburgh")
data_all <- bind_rows(map(cities, read_city))
raw_counts <- count(data_all, city)

# Rmd filtering_data chunk, plus the time_created cutoff (deviation 1).
data_filtered <-
  data_all %>%
  rename(pano_lat = panorama_lat, pano_lng = panorama_lng, pano_id = gsv_panorama_id) %>%
  filter(is.na(time_created) | time_created < CUTOFF) %>%  # NA: old DC rows predate the column
  filter(lat >= -90, lat <= 90, lng >= -180, lng <= 180, canvas_x > 0, canvas_y > 0,
         computation_method == 'depth', tutorial == FALSE, deleted == FALSE) %>%
  group_by(pano_id) %>%
  mutate(n = n()) %>%
  ungroup() %>%
  filter(n < MAX_LABELS_PER_PANO) %>%
  select(-n) %>%
  mutate(zoom = factor(zoom, levels = c(1, 2, 3))) %>%
  select(label_id, city, label_type, lat, lng, pano_lat, pano_lng, canvas_x, canvas_y,
         heading, zoom, sv_image_y, canvas_width, sv_image_x)  # keep what the models use

data_filtered_with_dist <-
  data_filtered %>%
  mutate(pano_dist = distHaversine(cbind(lng, lat), cbind(pano_lng, pano_lat))) %>%
  filter(pano_dist < MAX_DIST_FROM_PANO)

set.seed(666)
data_train <- data_filtered_with_dist %>% sample_frac(TRAINING_FRAC)
data_test  <- anti_join(data_filtered_with_dist, data_train, by = c('label_id', 'city'))

deg2rad <- function(d) d * pi / 180  # NISTunits::NISTdegTOradian

# heading_diff / label_heading on the training set (Rmd estimate4 chunk, vectorized).
add_heading_diff <- function(df) {
  df %>%
    mutate(label_heading = bearing(cbind(pano_lng, pano_lat), cbind(lng, lat)) %% 360,
           heading_diff = case_when(
             label_heading - heading > 180  ~ label_heading - heading - 360,
             label_heading - heading < -180 ~ label_heading - heading + 360,
             TRUE                           ~ label_heading - heading))
}
data_train_hd <- add_heading_diff(data_train)
data_test_hd  <- add_heading_diff(data_test)   # ground-truth heading_diff for error metrics

# --- Estimate 1: 10 m ahead, heading diff 0. (Rmd's crude flat-earth formula, kept verbatim.)
crude_latlng <- function(df, dist, hdiff = 0) {
  tibble(lat_est = df$pano_lat + (dist * cos(deg2rad(df$heading)) / 111111),
         lng_est = df$pano_lng + (dist * sin(deg2rad(df$heading)) /
                                    (111111 * cos(deg2rad(df$pano_lat)))))
}
est_err <- function(df, est) distHaversine(cbind(df$lng, df$lat), cbind(est$lng_est, est$lat_est))

err <- tibble(label_id = data_test_hd$label_id, city = data_test_hd$city)
e1 <- crude_latlng(data_test_hd, 10)
err$error_est1 <- est_err(data_test_hd, e1)
err$dist_error_est1    <- abs(data_test_hd$pano_dist - 10)
err$heading_error_est1 <- abs(data_test_hd$heading_diff - 0)

# --- Estimate 2: median training distance.
median_dist <- median(data_train$pano_dist)
e2 <- crude_latlng(data_test_hd, median_dist)
err$error_est2 <- est_err(data_test_hd, e2)
err$dist_error_est2    <- abs(data_test_hd$pano_dist - median_dist)
err$heading_error_est2 <- abs(data_test_hd$heading_diff - 0)

# --- Estimate 3: median training distance by label type.
dist_by_lab_type <-
  data_train %>% group_by(label_type) %>%
  summarise(med_pano_dist = median(pano_dist), .groups = 'drop') %>% deframe()
d3 <- dist_by_lab_type[as.character(data_test_hd$label_type)]
e3 <- crude_latlng(data_test_hd, d3)
err$error_est3 <- est_err(data_test_hd, e3)
err$dist_error_est3    <- abs(data_test_hd$pano_dist - d3)
err$heading_error_est3 <- abs(data_test_hd$heading_diff - 0)

# --- Estimate 4: multivariate lm, cbind(heading_diff, pano_dist) ~ canvas_y + sv_image_y.
mlm <- lm(cbind(heading_diff, pano_dist) ~ canvas_y + sv_image_y, data = data_train_hd)
pred_mlm <- predict(mlm, data_test_hd)
p4_h <- pred_mlm[, 'heading_diff']; p4_d <- pmax(0, pred_mlm[, 'pano_dist'])
dp4 <- destPoint(cbind(data_test_hd$pano_lng, data_test_hd$pano_lat),
                 data_test_hd$heading + p4_h, p4_d)
err$error_est4 <- distHaversine(cbind(data_test_hd$lng, data_test_hd$lat), dp4)
err$dist_error_est4    <- abs(data_test_hd$pano_dist - p4_d)
err$heading_error_est4 <- abs(data_test_hd$heading_diff - p4_h)

# --- Estimate 5: separate lms with zoom as a (factor) covariate.
lm_dist    <- lm(pano_dist ~ sv_image_y + canvas_y + zoom, data = data_train_hd)
lm_heading <- lm(heading_diff ~ canvas_x + zoom, data = data_train_hd)
p5_d <- pmax(0, predict(lm_dist, data_test_hd))
p5_h <- predict(lm_heading, data_test_hd)
dp5 <- destPoint(cbind(data_test_hd$pano_lng, data_test_hd$pano_lat),
                 data_test_hd$heading + p5_h, p5_d)
err$error_est5 <- distHaversine(cbind(data_test_hd$lng, data_test_hd$lat), dp5)
err$dist_error_est5    <- abs(data_test_hd$pano_dist - p5_d)
err$heading_error_est5 <- abs(data_test_hd$heading_diff - p5_h)

# --- Estimate 6: linear mixed effects with zoom as random intercept (lme4).
est6 <- tryCatch({
  library(lme4)
  lme_heading <- lmer(heading_diff ~ canvas_x + (1 | zoom), data = data_train_hd)
  lme_dist    <- lmer(pano_dist ~ canvas_y + sv_image_y + (1 | zoom), data = data_train_hd)
  p6_h <- predict(lme_heading, data_test_hd)
  p6_d <- pmax(0, predict(lme_dist, data_test_hd))
  dp6 <- destPoint(cbind(data_test_hd$pano_lng, data_test_hd$pano_lat),
                   data_test_hd$heading + p6_h, p6_d)
  err$error_est6         <- distHaversine(cbind(data_test_hd$lng, data_test_hd$lat), dp6)
  err$dist_error_est6    <- abs(data_test_hd$pano_dist - p6_d)
  err$heading_error_est6 <- abs(data_test_hd$heading_diff - p6_h)
  list(available = TRUE,
       heading = list(fixef = as.list(fixef(lme_heading)),
                      ranef_zoom = as.list(setNames(ranef(lme_heading)$zoom[[1]],
                                                    rownames(ranef(lme_heading)$zoom)))),
       dist = list(fixef = as.list(fixef(lme_dist)),
                   ranef_zoom = as.list(setNames(ranef(lme_dist)$zoom[[1]],
                                                 rownames(ranef(lme_dist)$zoom)))))
}, error = function(e) list(available = FALSE, error = conditionMessage(e)))

# --- Estimate 7: separate lms per zoom level (the winner).
fit7 <- function(train) {
  list(dist = lapply(1:3, function(z)
         as.list(coef(lm(pano_dist ~ sv_image_y + canvas_y, data = filter(train, zoom == z))))),
       heading = lapply(1:3, function(z)
         as.list(coef(lm(heading_diff ~ canvas_x, data = filter(train, zoom == z))))))
}
est7 <- fit7(data_train_hd)

p7_d <- rep(NA_real_, nrow(data_test_hd)); p7_h <- rep(NA_real_, nrow(data_test_hd))
for (z in 1:3) {
  i <- data_test_hd$zoom == z
  cd <- est7$dist[[z]]; ch <- est7$heading[[z]]
  p7_d[i] <- pmax(0, cd[["(Intercept)"]] + cd[["sv_image_y"]] * data_test_hd$sv_image_y[i] +
                     cd[["canvas_y"]] * data_test_hd$canvas_y[i])
  p7_h[i] <- ch[["(Intercept)"]] + ch[["canvas_x"]] * data_test_hd$canvas_x[i]
}
dp7 <- destPoint(cbind(data_test_hd$pano_lng, data_test_hd$pano_lat),
                 data_test_hd$heading + p7_h, p7_d)
err$error_est7 <- distHaversine(cbind(data_test_hd$lng, data_test_hd$lat), dp7)
err$dist_error_est7    <- abs(data_test_hd$pano_dist - p7_d)
err$heading_error_est7 <- abs(data_test_hd$heading_diff - p7_h)

# est7 refit on the full cleaned dataset: split-independent target for the Python tests.
est7_full <- fit7(add_heading_diff(data_filtered_with_dist %>%
                                     mutate(zoom = factor(zoom, levels = levels(data_train_hd$zoom)))))

# --- Summaries (Rmd summary_stats chunk).
ests <- names(err)[grepl("^error_est", names(err))]
summary_stats <- map_dfr(ests, function(cn) {
  v <- err[[cn]]
  tibble(estimate = cn, mean = mean(v), median = median(v), min = min(v), max = max(v), sd = sd(v))
}) %>% arrange(median)
heading_medians <- map_dbl(set_names(sub("error", "heading_error", ests)), ~ median(err[[.x]]))
dist_medians    <- map_dbl(set_names(sub("error", "dist_error", ests)),    ~ median(err[[.x]]))

# --- Fixtures.
write_csv(data_train %>% select(label_id, city) %>% arrange(city, label_id),
          file.path(fix_dir, "split_train.csv.gz"))
write_csv(data_test %>% select(label_id, city) %>% arrange(city, label_id),
          file.path(fix_dir, "split_test.csv.gz"))

fixtures <- list(
  meta = list(
    generated_by = "scripts/rerun-analysis.R",
    r_version = R.version.string,
    package_versions = list(readr = as.character(packageVersion("readr")),
                            dplyr = as.character(packageVersion("dplyr")),
                            geosphere = as.character(packageVersion("geosphere")),
                            lme4 = if (est6$available) as.character(packageVersion("lme4")) else NA),
    cutoff_utc = format(CUTOFF, "%Y-%m-%dT%H:%M:%SZ"),
    seed = 666, training_frac = TRAINING_FRAC,
    raw_rows_per_city = as.list(deframe(raw_counts)),
    rows_after_cleaning = nrow(data_filtered_with_dist),
    rows_train = nrow(data_train), rows_test = nrow(data_test)),
  est1 = list(dist = 10, heading_diff = 0),
  est2 = list(median_dist = median_dist),
  est3 = list(median_dist_by_label_type = as.list(dist_by_lab_type)),
  est4 = list(coefficients = apply(coef(mlm), 2, as.list)),
  est5 = list(dist = as.list(coef(lm_dist)), heading = as.list(coef(lm_heading))),
  est6 = est6,
  est7 = est7,
  est7_full = est7_full,
  error_stats = list(summary = summary_stats,
                     heading_error_medians = as.list(heading_medians),
                     dist_error_medians = as.list(dist_medians))
)
write_json(fixtures, file.path(fix_dir, "baseline.json"),
           auto_unbox = TRUE, digits = I(17), pretty = TRUE)

cat("\n=== Re-run complete ===\n")
cat(sprintf("Rows: raw %d -> cleaned %d (train %d / test %d)\n",
            nrow(data_all), nrow(data_filtered_with_dist), nrow(data_train), nrow(data_test)))
cat("\nTest-set error summary (m), sorted by median:\n")
print(as.data.frame(summary_stats), digits = 4)
cat("\nest7 coefficients (2021 published: dist z1 18.6051843/0.0138947/0.0011023, z2 20.8794248/0.0184087/0.0022135, z3 25.2472682/0.0264216/0.0011071;\n heading z1 -51.2401711/0.1443374, z2 -27.5267447/0.0784357, z3 -13.5675945/0.0396061):\n")
for (z in 1:3) {
  cd <- est7$dist[[z]]; ch <- est7$heading[[z]]
  cat(sprintf("zoom %d: dist %.7f + %.7f*sv_image_y + %.7f*canvas_y | heading %.7f + %.7f*canvas_x\n",
              z, cd[["(Intercept)"]], cd[["sv_image_y"]], cd[["canvas_y"]],
              ch[["(Intercept)"]], ch[["canvas_x"]]))
}
cat(sprintf("\nFixtures written to %s\n", normalizePath(fix_dir)))
