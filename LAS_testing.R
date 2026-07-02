library(tidyverse)
library(pracma)
library(data.table)
library(minpack.lm)
library(plotly)
library(mclust)
library(parallel)

tidy_conversion <- function(data) {
   x_axis <- data[, 1]
   data2 <- data |>
       rename(x_axis = V1) |>
       pivot_longer(
           cols = paste0("V", seq(2:length(data)) + 1),
           names_to = "id", 
           values_to = "intensity"
       ) |>
       mutate(id = as.numeric(sub("V", "", id)) - 1)
}

find_peak_locations <- function(raw, cl) {
    data <- raw[raw$V1 > 375 & raw$V1 < 420, ]
    x_axis <- data$V1
    
    e2g_idx <- which(x_axis > 375 & x_axis < 395)
    a1g_idx <- which(x_axis > 395 & x_axis < 420)
    results <- vector("list", ncol(data) - 1)
    
    results <- parLapply(cl, 2:ncol(data), function(i) {
        intensity <- data[[i]]
        intensity <- intensity / max(raw[[i]], na.rm = TRUE) * 1000
        
        peak1 <- e2g_idx[which.max(intensity[e2g_idx])]
        peak2 <- a1g_idx[which.max(intensity[a1g_idx])]
        
        print(paste0("Spectrum ", (i - 1), " processed."))
        
        tibble::tibble(
            id = i - 1,
            x_axis1 = x_axis[peak1],
            intensity1 = intensity[peak1],
            x_axis2 = x_axis[peak2],
            intensity2 = intensity[peak2]
        )
    })
    
    peak_locations <- dplyr::bind_rows(results)
    peak_locations
}

auto_gaussian_summary <- function(raw, peak_locations, cl) {
    data <- raw[raw$V1 > 375 & raw$V1 < 420, ]
    x <- data$V1
    
    results <- parLapply(cl, seq_len(nrow(peak_locations)), function(i) {
        gaussian <- function(x, A, mu, sigma) {
            A * exp(-(x - mu)^2 / (2 * sigma^2))
        }
        double_gaussian <- function(x, A1, mu1, sigma1, A2, mu2, sigma2, C) {
            gaussian(x, A1, mu1, sigma1) + gaussian(x, A2, mu2, sigma2) + C
        }
        
        spectrum_id <- peak_locations$id[i]
        
        y <- data[[spectrum_id + 1]]
        spectrum_max <- max(raw[[spectrum_id + 1]], na.rm = TRUE)
        y <- y / spectrum_max * 1000
        
        A1_guess <- peak_locations$intensity1[i]
        A2_guess <- peak_locations$intensity2[i]
        mu1_guess <- peak_locations$x_axis1[i]
        mu2_guess <- peak_locations$x_axis2[i]
        
        error_return <- tibble::tibble(
            id = spectrum_id, mu1 = 0, mu2 = 0, fwhm1 = 0, fwhm2 = 0,
            A1 = 0, A2 = 0, area1 = 0, area2 = 0, area_ratio = 0,
            snr = 0, rmse = 0, r_squared = 0, diff_fit = 0
        )
        
        fit <- tryCatch({
            nlsLM(y ~ double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, C),
                  start = list(
                      A1 = A1_guess, mu1 = mu1_guess, sigma1 = 3,
                      A2 = A2_guess, mu2 = mu2_guess, sigma2 = 3,
                      C = min(y)
                  ),
                  lower = c(0, 370, 0.5, 0, 390, 0.5, 0),
                  upper = c(1500, 400, 20, 1500, 430, 20, 1500)
            )
        },
        error = function(e) {
            cat("Spectrum:", spectrum_id, "\n", e$message, "\n\n")
            NULL
        })
        
        if (is.null(fit)) {
            results[[i]] <- error_return
            next
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
        
        if (r_squared < 0.90) {
            results[[i]] <- error_return
            print("r_squared below 90%")
            next
        }
        
        noise <- sd(residuals)
        snr <- ifelse(noise == 0, Inf, max(y) / noise)
        
        results[[i]] <- tibble(
            id = spectrum_id, mu1 = p["mu1"], mu2 = p["mu2"],
            fwhm1 = fwhm1, fwhm2 = fwhm2, A1 = p["A1"], A2 = p["A2"],
            area1 = area1, area2 = area2, area_ratio = area_ratio,
            snr = snr, rmse = rmse, r_squared = r_squared,
            diff_fit = abs(p["mu2"] - p["mu1"])
        )
    })
    
    dplyr::bind_rows(results)
}

num_cores <- detectCores(logical = FALSE) - 1
cl <- makeCluster(num_cores, type = "PSOCK")

size <- 300
file_path <- paste0(
    "data/default LAS/", 
    size, "x", size, 
    "/Large Area Scan.csv"
    )

compute_time <- round(0.00588271 * size ^ 2 + 2.21832, 2)
paste0("Time to Compute: ", compute_time %/% 60, ":", (compute_time %% 60))
raw <- fread(file_path, header = FALSE)
data <- raw

peak_summary <- find_peak_locations(data, cl)
gaussian_results <- auto_gaussian_summary(data, peak_summary, cl)
stopCluster(cl)

peak_summary <- peak_summary |>
    mutate(
        diff_peak = abs(x_axis1 - x_axis2),
        intensity_ratio = intensity1 / intensity2,
        intensity_ratio = ifelse(intensity_ratio > 1, intensity_ratio, 1/intensity_ratio)
    ) |>
    merge(gaussian_results, by = "id")

heatmap_df <- peak_summary |>
    dplyr::mutate(
        x = ((id - 1) %% size) + 1,
        y = ((id - 1) %/% size) + 1,
        #curve = intensity1 > 730 & intensity2 > 730,
        #diff_peak = ifelse(curve, diff_peak, 15),
        #diff_peak = ifelse(diff_peak > 26 | diff_peak < 18, 15, diff_peak),
        #intensity_ratio = ifelse(curve, ratio, 0.9),
        intensity_ratio = ifelse(intensity_ratio > 1.25, 1.25, intensity_ratio)
        #mu1 = ifelse(curve, mu1, 0),
        #mu2 = ifelse(curve, mu2, 0),
        #fwhm1 = ifelse(curve, fwhm1, 0),
        #fwhm2 = ifelse(curve, fwhm2, 0),
        ##A1 = ifelse(curve, A1, 0),
        #A2 = ifelse(curve, A2, 0),
        #area1 = ifelse(curve, area1, 0),
        #area2 = ifelse(curve, area2, 0),
        #area_ratio = ifelse(curve, area_ratio, 0),
        #snr = ifelse(curve, snr, 0),
        #rmse = ifelse(curve, rmse, 0),
        #r_squared = ifelse(curve, r_squared, 0),
        #diff_fit = ifelse(curve, diff_fit, 0)
    ) |>
    dplyr::select(
        id, x, y, x_axis1, x_axis2, diff_peak, mu1, mu2, diff_fit, 
        intensity1, intensity2, intensity_ratio, A1, A2,  fwhm1, fwhm2, 
        area1, area2, area_ratio, snr, rmse, r_squared)#, curve)

### PCA ANALYSIS ###
pca_data <- data[data$V1 > 375 & data$V1 < 420, ]
spectra_matrix <- t(as.matrix(pca_data[ , -1]))
spectra_scaled <- scale(spectra_matrix)
pca <- prcomp(spectra_scaled, center = TRUE, scale. = TRUE)

#plot(cumsum(pca$sdev^2 / sum(pca$sdev^2)), type = "b", 
#     xlab = "PC", ylab = "Cumulative Variance Explained")

pca_scores <- as.data.frame(pca$x[, 1:5])
pca_scores$id <- seq_len(nrow(pca_scores))
heatmap_df <- cbind(heatmap_df, pca_scores[, 1:5])

kmeans_vars <- heatmap_df |>
    dplyr::select(
        diff_peak, diff_fit, mu1, mu2, intensity_ratio, A1, A2, fwhm1, fwhm2, 
        area1, area2, area_ratio, snr, rmse, r_squared, PC1, PC2, PC3, PC4, PC5
        ) |>
    scale()

cluster_num <- 4
clustering_results <- kmeans(
    kmeans_vars, 
    centers = cluster_num)

ordering <- order(clustering_results$centers[, 1])
mapping <- setNames(1:cluster_num, ordering)
clustering_results$cluster <- mapping[as.character(clustering_results$cluster)]

heatmap_df$cluster <- (clustering_results$cluster - 1) * 6.15

p <- ggplot(heatmap_df, aes(x = x, y = y, fill = cluster)) +
    geom_tile() +
    coord_equal() +
    scale_y_reverse() +
    scale_fill_gradientn(colors = c("lightblue", "yellow", "red")) + 
    labs(fill = "Clustering") + 
    theme_bw()
p
ggplotly(p)

write.csv(heatmap_df, file = "../paraview_data/analysis_results.csv", row.names = FALSE)

ggplotly(ggplot(data, aes(x = V1, y = V41)) + geom_line() + theme_bw())

# --------- #

large_area_features <- heatmap_df
prediction <- predict(
    lda_model,
    newdata = large_area_features
)
large_area_features$cluster <- prediction$class
large_area_features <- large_area_features |>
    mutate(
        cluster = ifelse(cluster == "background", 0, ifelse(cluster == "monolayer", 1, 2))
    )

p <- ggplot(large_area_features, aes(x = x, y = y, fill = cluster)) +
    geom_tile() +
    coord_equal() +
    scale_y_reverse() +
    scale_fill_gradientn(colors = c("lightblue", "yellow", "red")) + 
    labs(fill = "Clustering") + 
    theme_bw()
p
ggplotly(p)