library(MASS)
library(mclust)
library(plotly)
library(tidyverse)
library(pracma)
library(data.table)
library(minpack.lm)
library(parallel)
library(glmnet)
library(randomForest)

normalize_data <- function(raw) {
    cols <- 2:ncol(raw)
    mat <- as.matrix(raw[, ..cols])
    maxes <- apply(mat, 2, max, na.rm = TRUE)
    mat <- sweep(mat, 2, maxes, "/") * 1000
    raw[, (cols) := as.data.table(mat)]
    return(raw)
}

find_peak_locations <- function(raw, cl) {
    data <- raw[raw$V1 > 375 & raw$V1 < 420, ]
    x_axis <- data$V1
    
    # Matrix of spectra (rows = Raman shifts, columns = spectra)
    spectra <- as.matrix(data[, -1])
    max_vals <- apply(as.matrix(raw[, -1]), 2, max, na.rm = TRUE)
    
    e2g <- spectra[x_axis > 375 & x_axis < 395, , drop = FALSE]
    a1g <- spectra[x_axis > 395 & x_axis < 420, , drop = FALSE]
    peak1 <- max.col(t(e2g), ties.method = "first")
    peak2 <- max.col(t(a1g), ties.method = "first")
    e2g_rows <- which(x_axis > 375 & x_axis < 395)
    a1g_rows <- which(x_axis > 395 & x_axis < 420)
    peak1_global <- e2g_rows[peak1]
    peak2_global <- a1g_rows[peak2]
    
    peak_locations <- data.frame(
        id = seq_len(ncol(spectra)),
        x_axis1 = x_axis[peak1_global],
        intensity1 = spectra[cbind(peak1_global, seq_len(ncol(spectra)))],
        x_axis2 = x_axis[peak2_global],
        intensity2 = spectra[cbind(peak2_global, seq_len(ncol(spectra)))]
    )
}

gaussian <- function(x, A, mu, sigma) {
    A * exp(-(x - mu) ^ 2 / (2 * sigma ^ 2))
}
double_gaussian <- function(x, A1, mu1, sigma1, A2, mu2, sigma2, C) {
    gaussian(x, A1, mu1, sigma1) + gaussian(x, A2, mu2, sigma2) + C
}

auto_gaussian_summary <- function(raw, peak_locations, cl) {
    data <- raw[raw$V1 > 375 & raw$V1 < 420, ]
    x <- data$V1
    
    chunks <- split(
        seq_len(nrow(peak_locations)),
        cut(seq_len(nrow(peak_locations)), length(cl), labels = FALSE)
    )
    
    results <- parLapply(cl, seq_len(nrow(peak_locations)), function(i) {
        spectrum_id <- peak_locations$id[i]
        
        y <- data[[spectrum_id + 1]]
        
        A1_guess <- peak_locations$intensity1[i]
        A2_guess <- peak_locations$intensity2[i]
        mu1_guess <- peak_locations$x_axis1[i]
        mu2_guess <- peak_locations$x_axis2[i]
        
        error_return <- tibble::tibble(
            id = spectrum_id, mu1 = 0, mu2 = 0, fwhm1 = 0, fwhm2 = 0,
            A1 = 0, A2 = 0, area1 = 0, area2 = 0, area_ratio = 0,
            snr = 0, rmse = 0, r_squared = 0, diff_fit = 0, status = "failed"
        )
        
        fit <- tryCatch({
            nlsLM(y ~ double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, C),
                  start = list(
                      A1 = A1_guess, mu1 = mu1_guess, sigma1 = 3,
                      A2 = A2_guess, mu2 = mu2_guess, sigma2 = 3,
                      C = min(y)
                  ),
                  lower = c(0, 370, 0.5, 0, 390, 0.5, 0),
                  upper = c(1100, 400, 20, 1100, 430, 20, 1100)
            )
        },
        error = function(e) {
            message("Spectrum ", spectrum_id, ": ", e$message)
            NULL
        })
        
        if (is.null(fit)) {
            error_return$status <- "no fit"
            return(error_return)
        }
        
        p <- coef(fit)
        fwhm1 <- 2.35482 * p["sigma1"]
        fwhm2 <- 2.35482 * p["sigma2"]
        
        area1 <- p["A1"] * p["sigma1"] * sqrt(2 * pi)
        area2 <- p["A2"] * p["sigma2"] * sqrt(2 * pi)
        area_ratio <- area1 / area2
        
        fitted_y <- double_gaussian(
            x, p["A1"], p["mu1"], p["sigma1"], 
            p["A2"], p["mu2"], p["sigma2"], p["C"]
        )
        residuals <- y - fitted_y
        rmse <- sqrt(mean(residuals^2))
        
        ss_res <- sum(residuals^2)
        ss_tot <- sum((y - mean(y))^2)
        r_squared <- 1 - ss_res / ss_tot
        
        if (r_squared < 0.85) {
            error_return$status <- "r^2 too low"
            error_return$r_squared <- r_squared
            return(error_return)
        }
        
        noise <- sd(residuals)
        snr <- ifelse(noise == 0, Inf, max(y) / noise)
        
        tibble(
            id = spectrum_id, mu1 = p["mu1"], mu2 = p["mu2"],
            fwhm1 = fwhm1, fwhm2 = fwhm2, A1 = p["A1"], A2 = p["A2"],
            area1 = area1, area2 = area2, area_ratio = area_ratio,
            snr = snr, rmse = rmse, r_squared = r_squared,
            diff_fit = abs(p["mu2"] - p["mu1"]), status = "success"
        )
    })
    
    dplyr::bind_rows(results)
}

process_spectrum <- function(file, id, cl){
    raw <- fread(file, header = FALSE)
    data <- normalize_data(raw)
    
    peak <- find_peak_locations(data, cl)
    fit  <- auto_gaussian_summary(data, peak, cl)
    result <- peak |>
        dplyr::mutate(
            diff_peak = abs(x_axis1 - x_axis2),
            intensity_ratio = intensity1 / intensity2,
            intensity_ratio = ifelse(intensity_ratio > 1, intensity_ratio, 1/intensity_ratio)
        ) |>
        left_join(fit, by = "id")
    
    result$id <- id
    result$file <- file
    return(result)
}

### MODEL CREATION ###

filepath <- c(
    "data/training_data/background/07072025_1.txt",
    "data/training_data/background/07072025_2.txt",
    "data/training_data/background/07072025_3.txt",
    "data/training_data/background/07072025_4.txt",
    "data/training_data/background/07072025_5.txt",
    "data/training_data/background/07072025_6.txt",
    "data/training_data/monolayer/07102025_1.txt",
    "data/training_data/monolayer/07102025_2.txt",
    "data/training_data/monolayer/07102025_3.txt",
    "data/training_data/monolayer/07102025_4.txt",
    "data/training_data/monolayer/07102025_5.txt",
    "data/training_data/monolayer/07102025_6.txt",
    "data/training_data/monolayer/07102025_7.txt",
    "data/training_data/monolayer/07102025_8.txt",
    "data/training_data/monolayer/07102025_9.txt",
    "data/training_data/monolayer/07102025_10.txt",
    "data/training_data/monolayer/07102025_11.txt",
    "data/training_data/monolayer/07102025_12.txt",
    "data/training_data/monolayer/07102025_13.txt",
    "data/training_data/monolayer/07152025_1.txt",
    "data/training_data/monolayer/07152025_2.txt",
    "data/training_data/monolayer/07152025_3.txt",
    "data/training_data/monolayer/07152025_4.txt",
    "data/training_data/monolayer/07152025_5.txt",
    "data/training_data/monolayer/07152025_6.txt",
    "data/training_data/monolayer/07152025_8.txt",
    "data/training_data/monolayer/07152025_9.txt",
    "data/training_data/monolayer/07152025_10.txt",
    "data/training_data/monolayer/07172025_1.txt",
    "data/training_data/monolayer/08132025_1.txt",
    "data/training_data/monolayer/08132025_2.txt",
    "data/training_data/monolayer/08132025_3.txt",
    "data/training_data/monolayer/08132025_4.txt",
    "data/training_data/monolayer/08132025_5.txt",
    "data/training_data/bilayer/07152025_1.txt",
    "data/training_data/bilayer/07152025_2.txt",
    "data/training_data/bilayer/07152025_3.txt",
    "data/training_data/bilayer/07152025_4.txt",
    "data/training_data/bilayer/08142025_1.txt",
    "data/training_data/bilayer/08142025_2.txt",
    "data/training_data/bilayer/08142025_3.txt"
)

# PCA ON RAW SPECTRA

num_cores <- detectCores(logical = FALSE) - 1
cl <- makeCluster(num_cores, type = "PSOCK")

# cores || user || system || elapsed
# 1 || 15 || 3 || 433
# 2 || 18 || 8 || 299
# 4 || 10 || 6 || 247
# 7 || 17 || 23 || 200

clusterEvalQ(cl, {
    library(minpack.lm) 
    library(tibble)
    })
clusterExport(cl, c("gaussian", "double_gaussian"))

# -------------------------------- MODEL SETUP ------------------------------- #
spectra <- lapply(filepath, fread)
results <- bind_rows(lapply(seq_along(filepath), function(i) {
    process_spectrum(filepath[i], i, cl)
}))

labels <- as.factor(basename(dirname(filepath)))
results$Layer <- labels
map_vector <- c("background" = 0, "monolayer" = 0.7, "bilayer" = 1.5)
results$thickness <- map_vector[as.character(results$Layer)]

feature_table <- results |>
    dplyr::select(
        Layer, thickness, x_axis1, x_axis2, diff_peak, intensity_ratio, 
        mu1, mu2, fwhm1, fwhm2, A1, A2, area1, area2, area_ratio, snr, rmse,
        r_squared, diff_fit
    ) |> 
    dplyr::select(Layer, thickness, diff_peak, A1, A2, area_ratio, fwhm1, fwhm2, diff_fit)

scaled_features <- scale(feature_table[, -c(1, 2)])

center <- attr(scaled_features,"scaled:center")
scale  <- attr(scaled_features,"scaled:scale")

scaled_features <- as.data.frame(scaled_features)
scaled_features$Layer <- feature_table$Layer
scaled_features$thickness <- feature_table$thickness

### LINEAR DISCRIMINATORY ANALYSIS ###
lda_model_benchmark <- lda(
    thickness ~ (diff_peak + A1 + A2 + area_ratio + fwhm1 + fwhm2 + diff_fit),
    data = scaled_features, CV = TRUE
)

table(Actual = feature_table$thickness, Predicted = lda_model_benchmark$class)
lda_model <- lda(
    Layer ~ (diff_peak + A1 + A2 + area_ratio + fwhm1 + fwhm2 + diff_fit),
    data = scaled_features
)

### LINEAR REGRESSION ###
linear_model <- lm(
    thickness ~ (diff_peak + A1 + A2 + area_ratio + fwhm1 + fwhm2 + diff_fit),
    data = scaled_features)

### RIDGE REGRESSION ###
X <- as.matrix(scaled_features[1:7])
y <- scaled_features$thickness
ridge_cv_model <- cv.glmnet(X, y, alpha = 0)
best_lambda <- ridge_cv_model$lambda.min
print(best_lambda)
plot(ridge_cv_model)
ridge_model <- glmnet(X, y, alpha = 0, lambda = best_lambda)
coef(ridge_model)

### RANDOM FOREST REGRESSION ###
forest_model <- randomForest(
    thickness ~ (diff_peak + A1 + A2 + area_ratio + fwhm1 + fwhm2 + diff_fit),
    data = scaled_features, ntree = 500, mtry = 2, importance = TRUE
)


### SUPPORT VECTOR REGRESSION ###

######### # NEURAL?! ########## 


### LARGE AREA SCAN PROCESSING ###

size <- 300
file_path <- paste0(
    "data/default LAS/", 
    size, "x", size, 
    "/Large Area Scan.csv"
)

compute_time <- round(0.00588271 * size ^ 2 + 2.21832, 2)
paste0("Time to Compute: ", compute_time %/% 60, ":", (compute_time %% 60))
raw <- fread(file_path, header = FALSE)
data <- normalize_data(raw)

peak_summary <- find_peak_locations(data, cl)
gaussian_results <- auto_gaussian_summary(data, peak_summary, cl)
stopCluster(cl)

peak_summary <- peak_summary |>
    mutate(
        diff_peak = abs(x_axis1 - x_axis2),
        intensity_ratio = intensity1 / intensity2,
        intensity_ratio = ifelse(
            intensity_ratio > 1, 
            intensity_ratio, 
            1 / intensity_ratio
            )
    ) |>
    merge(gaussian_results, by = "id")

heatmap_df <- peak_summary |>
    dplyr::mutate(
        x = ((id - 1) %% size) + 1,
        y = ((id - 1) %/% size) + 1
        ) |>
    dplyr::select(
        id, x, y, x_axis1, x_axis2, diff_peak, mu1, mu2, diff_fit, 
        intensity1, intensity2, intensity_ratio, A1, A2,  fwhm1, fwhm2, 
        area1, area2, area_ratio, snr, rmse, r_squared)

### LDA MODEL APPLICATION ###
large_area_features <- heatmap_df |>
    dplyr::select(diff_peak, A1, A2, area_ratio, fwhm1, fwhm2, diff_fit)
large_scaled <- scale(
    large_area_features,
    center = center,
    scale = scale
)

lda_pred <- predict(lda_model, newdata = as.data.frame(large_scaled))
lm_pred <- predict(linear_model, newdata = as.data.frame(large_scaled))
rr_pred <- predict(ridge_model, s = best_lambda, newx = large_scaled)
rf_pred <- predict(forest_model, newdata = as.data.frame(large_scaled))

large_area_features$cluster_lda <- lda_pred$class
large_area_features$thickness_lda <- map_vector[as.character(lda_pred$class)] * 5
large_area_features$thickness_lm <- as.numeric(lm_pred * 5)
large_area_features$thickness_rr <- as.numeric(as.vector(rr_pred) * 5)
large_area_features$thickness_rf <- rf_pred * 5
large_area_features$x <- heatmap_df$x
large_area_features$y <- heatmap_df$y
large_area_features$height <- large_area_features$thickness_lm

lm_p <- ggplot(large_area_features, aes(x = x, y = y, fill = thickness_lm)) +
    geom_tile() +
    coord_equal() +
    scale_y_reverse() +
    scale_fill_gradientn(
        colors = c("purple", "blue", "orange", "red"),
        limits = c(-25, 25), 
        oob = squish, 
        breaks = c(-25, 0, 2.5, 5, 25),
        labels = c("≤ -5", "0", "0.5", "1", "5")
    ) +
    labs(fill = "Clustering") + 
    theme_bw()
ggplotly(lm_p)

lda_p <- ggplot(large_area_features, aes(x = x, y = y, fill = thickness_lda)) +
    geom_tile() +
    coord_equal() +
    scale_y_reverse() +
    scale_fill_gradientn(colors = c("lightblue", "yellow", "red")) + 
    labs(fill = "Clustering") + 
    theme_bw()
ggplotly(lda_p)

rr_p <- ggplot(large_area_features, aes(x = x, y = y, fill = thickness_rr)) +
    geom_tile() +
    coord_equal() +
    scale_y_reverse() +
    scale_fill_gradientn(
        colors = c("purple", "blue", "orange", "red"),
        limits = c(-0.05, 10), 
        oob = squish, 
        breaks = c(-0.05, 0, 2.5, 5, 10),
        labels = c("≤ 0", "0", "0.5", "1", "2")
    ) +
    labs(fill = "Clustering") + 
    theme_bw()
ggplotly(rr_p)

rf_p <- ggplot(large_area_features, aes(x = x, y = y, fill = thickness_rf)) +
    geom_tile() +
    coord_equal() +
    scale_y_reverse() +
    scale_fill_gradientn(colors = c("lightblue", "yellow", "red")) + 
    labs(fill = "Clustering") + 
    theme_bw()
ggplotly(rf_p)

p <- ggplot(a, aes(x = x, y = y, fill = goofy_mean)) +
    geom_tile() +
    coord_equal() +
    scale_y_reverse() +
    scale_fill_gradientn(
        colors = c("purple", "blue", "orange", "red"),
        limits = c(-25, 25), 
        oob = squish, 
        breaks = c(-25, 0, 2.5, 5, 25),
        labels = c("≤ -5", "0", "0.5", "1", "5")
    ) +
    labs(fill = "Clustering") + 
    theme_bw()
ggplotly(p)

write.csv(large_area_features, file = "../paraview_data/analysis_results.csv", row.names = FALSE)
