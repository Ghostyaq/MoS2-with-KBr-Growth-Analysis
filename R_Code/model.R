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
library(e1071)
library(pls)
library(scales)

source("R_Code/functions.R")

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

nboot <- 1000
num_cores <- detectCores(logical = FALSE) - 1
cl <- makeCluster(num_cores, type = "PSOCK")
viz_scale <- 1

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
map_vector <- c("background" = 0, "monolayer" = 0.7, "bilayer" = 2.02)
results$thickness <- map_vector[as.character(results$Layer)]

feature_table <- results |>
    dplyr::select(
        Layer, thickness, x_axis1, x_axis2, diff_peak, intensity_ratio, 
        mu1, mu2, fwhm1, fwhm2, A1, A2, area1, area2, area_ratio, snr, rmse,
        r_squared, diff_fit
    )# |> 
    #dplyr::select(Layer, thickness, diff_peak, A1, A2, area_ratio, fwhm1, fwhm2, diff_fit)

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
svr_model <- svm(
    thickness ~ (diff_peak + A1 + A2 + area_ratio + fwhm1 + fwhm2 + diff_fit),
    data = scaled_features, type = "eps-regression", kernel = "radial"
)

### PARTIAL LINEAR REGRESSION ###
plsr_model <- plsr(
    thickness ~ 
        diff_peak + intensity_ratio + mu1 + mu2 + fwhm1 + fwhm2 + A1 + A2 + 
        area1 + area2 + area_ratio + snr + rmse + diff_fit,
    data = scaled_features,
    validation = "LOO",
    scale = FALSE
)
validationplot(plsr_model, val.type = "RMSEP")

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
    dplyr::select(
       x_axis1, x_axis2, diff_peak, intensity_ratio, 
        mu1, mu2, fwhm1, fwhm2, A1, A2, area1, area2, area_ratio, snr, rmse,
        r_squared, diff_fit
    )
large_scaled <- scale(
    large_area_features,
    center = center,
    scale = scale
)[, c(3, 7:10, 13, 17)]
large_scaled_plsr <- scale(
    large_area_features,
    center = center,
    scale = scale
)

lda_pred <- predict(lda_model, newdata = as.data.frame(large_scaled))
lm_pred <- predict(linear_model, newdata = as.data.frame(large_scaled))
rr_pred <- predict(ridge_model, s = best_lambda, newx = large_scaled)
rf_pred <- predict(forest_model, newdata = as.data.frame(large_scaled))
vr_pred <- predict(svr_model, newdata = as.data.frame(large_scaled))
pl_pred <- predict(plsr_model, newdata = as.data.frame(large_scaled_plsr), ncomp = 6)
boot_predictions <- matrix(NA, nrow = nrow(large_scaled), ncol = nboot)

set.seed(123)
for (i in 1:nboot) {
    back_features <- filter(scaled_features, Layer == "background")
    mono_features <- filter(scaled_features, Layer == "monolayer")
    bila_features <- filter(scaled_features, Layer == "bilayer")
    
    boot_index_back <- sample(seq_len(nrow(back_features)), replace = TRUE)
    boot_index_mono <- sample(seq_len(nrow(mono_features)), replace = TRUE)
    boot_index_bila <- sample(seq_len(nrow(bila_features)), replace = TRUE)
    
    boot_train_back <- back_features[boot_index_back, ]
    boot_train_mono <- mono_features[boot_index_mono, ]
    boot_train_bila <- bila_features[boot_index_bila, ]
    boot_train <- rbind(boot_train_back, boot_train_mono, boot_train_bila)
    
    model <- plsr(
        thickness ~ 
            diff_peak + intensity_ratio + mu1 + mu2 + fwhm1 + fwhm2 + A1 + A2 + 
            area1 + area2 + area_ratio + snr + rmse + diff_fit,
        data = boot_train, validation = "none", scale = FALSE, ncomp = 3
    )
    
    boot_predictions[, i] <- as.vector(
        predict(model, newdata = as.data.frame(large_scaled_plsr), ncomp = 3)
        )
}

mean_prediction <- rowMeans(boot_predictions)
sd_prediction <- apply(boot_predictions, 1, sd)
lower95 <- apply(boot_predictions, 1, quantile, probs = 0.025)
upper95 <- apply(boot_predictions, 1, quantile, probs = 0.975)

large_area_features$cluster_lda <- lda_pred$class
large_area_features$thickness_lda <- map_vector[as.character(lda_pred$class)] * viz_scale 
large_area_features$thickness_lm <- as.numeric(lm_pred) * viz_scale
large_area_features$thickness_rr <- as.numeric(as.vector(rr_pred)) * viz_scale
large_area_features$thickness_rf <- rf_pred * viz_scale
large_area_features$thickness_vr <- as.numeric(vr_pred) * viz_scale
large_area_features$thickness_pl <- as.numeric(pl_pred) * viz_scale
large_area_features$mean_thickness <- mean_prediction * viz_scale
large_area_features$uncertainty <- sd_prediction * viz_scale
large_area_features$lower95 <- lower95 * viz_scale
large_area_features$upper95 <- upper95 * viz_scale
large_area_features$x <- heatmap_df$x
large_area_features$y <- heatmap_df$y
large_area_features <- large_area_features |>
    mutate(
        height = (
            thickness_lda + thickness_lm + thickness_rr + 
            thickness_rf + thickness_rf + thickness_pl
            ) / 6
    )

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

vr_p <- ggplot(large_area_features, aes(x = x, y = y, fill = thickness_vr)) +
    geom_tile() +
    coord_equal() +
    scale_y_reverse() +
    scale_fill_gradientn(colors = c("lightblue", "yellow", "red")) + 
    labs(fill = "Clustering") + 
    theme_bw()
ggplotly(vr_p)

pl_p <- ggplot(large_area_features, aes(x = x, y = y, fill = mean_thickness)) +
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
ggplotly(pl_p)

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
