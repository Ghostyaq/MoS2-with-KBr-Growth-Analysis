rm(list = ls())

load("data/Rdata/benchmark_cv_full.RData")

final_results_df <- arrange(final_results_df, desc(R2))

raw <- final_results_df$Variables_Used

data_500 <- raw[1:500]
data_1000 <- raw[1:1000]
data_5000 <- raw[1:5000]
data_10000 <- raw[1:10000]

variables <- c(
    "x_axis1", "x_axis2", "diff_peak", "mu1", "mu2", "diff_fit", 
    "intensity_ratio", "A1", "A2",  "fwhm1", "fwhm2", "area1", 
    "area2", "area_ratio", "snr", "rmse", "r_squared"
    )

x <- variables
y_500 <- sapply(variables, function(x) sum(grepl(x, data_500)))
y_1000 <- sapply(variables, function(x) sum(grepl(x, data_1000)))
y_5000 <- sapply(variables, function(x) sum(grepl(x, data_5000)))
y_10000 <- sapply(variables, function(x) sum(grepl(x, data_10000)))

mh <- data.frame(
    variables = x, count_500 = y_500, count_1000 = y_1000, 
    count_5000 = y_5000, count_10000 = y_10000
    ) |> 
    arrange(desc(count_5000)) |>
    mutate(
        `Top 500` = count_500 / 500 * 100,
        `Top 1000` = count_1000 / 1000 * 100,
        `Top 5000` = count_5000 / 5000 * 100,
        `Top 10000` = count_10000 / 10000 * 100
    ) |>
    select(variables, `Top 500`, `Top 1000`, `Top 5000`, `Top 10000`)

rownames(mh) <- NULL
