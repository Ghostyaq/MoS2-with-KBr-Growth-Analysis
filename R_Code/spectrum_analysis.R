library(MASS)
library(plotly)
library(tidyverse)
library(pracma)
library(data.table)
library(minpack.lm)

# 6, 24, 8

find_peak_locations <- function(raw) {
    data <- raw[raw$V1 > 365 & raw$V1 < 430, ]
    x_axis <- data$V1
    
    e2g_idx <- which(x_axis > 375 & x_axis < 395)
    a1g_idx <- which(x_axis > 395 & x_axis < 420)
    results <- vector("list", ncol(data) - 1)
    
    for (i in 2:ncol(data)) {
        intensity <- data[[i]]
        intensity <- intensity / max(raw[[i]], na.rm = TRUE) * 1000
        
        peak1 <- e2g_idx[which.max(intensity[e2g_idx])]
        peak2 <- a1g_idx[which.max(intensity[a1g_idx])]
        
        results[[i - 1]] <- tibble(
            id = i - 1,
            x_axis1 = x_axis[peak1],
            intensity1 = intensity[peak1],
            x_axis2 = x_axis[peak2],
            intensity2 = intensity[peak2]
        )
    }
    peak_locations <- bind_rows(results)
    peak_locations
}

gaussian <- function(x, A, mu, sigma) {
    A * exp(-(x - mu) ^ 2 / (2 * sigma ^ 2))
}

double_gaussian <- function(x, A1, mu1, sigma1, A2, mu2, sigma2, C) {
    gaussian(x, A1, mu1, sigma1) + gaussian(x, A2, mu2, sigma2) + C
}

auto_gaussian_summary <- function(raw, peak_locations) {
    data <- raw[raw$V1 > 365 & raw$V1 < 430, ]
    x <- data$V1
    results <- vector("list", nrow(peak_locations))
    
    for (i in seq_len(nrow(peak_locations))) {
        spectrum_id <- peak_locations$id[i]
        y <- data[[spectrum_id + 1]]
        spectrum_max <- max(raw[[spectrum_id + 1]], na.rm = TRUE)
        y <- y / spectrum_max * 1000
        
        A1_guess <- peak_locations$intensity1[i] / spectrum_max * 1000
        A2_guess <- peak_locations$intensity2[i] / spectrum_max * 1000
        mu1_guess <- peak_locations$x_axis1[i]
        mu2_guess <- peak_locations$x_axis2[i]
        
        error_return <- tibble(
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
        
        if (r_squared < 0.85) {
            results[[i]] <- error_return
            print("r_squared below 85%")
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
    }
    bind_rows(results)
}

process_spectrum <- function(file, id){
    raw <- fread(file, header = FALSE)
    
    peak <- find_peak_locations(raw)
    fit  <- auto_gaussian_summary(raw, peak)
    result <- peak |>
        mutate(
            diff_peak = abs(x_axis1 - x_axis2),
            intensity_ratio = intensity1 / intensity2,
            intensity_ratio = ifelse(intensity_ratio > 1, intensity_ratio, 1/intensity_ratio)
        ) |>
        left_join(fit, by = "id")
    
    result$id <- id
    result$file <- file
    return(result)
}

pca_explained_var <- function(pca) {
    explained_var <- pca$sdev ^ 2
    explained_var_ratio <- explained_var / sum(explained_var)
    
    variance_df <- data.frame(
        PC = factor(
            paste0("PC", seq_along(explained_var_ratio)),
            levels = paste0("PC", seq_along(explained_var_ratio))),
        Variance = explained_var_ratio * 100
    )
    
    variance_df$Cumulative <- cumsum(variance_df$Variance)
    
    ggplot(variance_df, aes(x = seq_along(Cumulative), y = Cumulative)) +
        geom_line() +
        geom_point(size = 2) +
        scale_x_continuous(breaks = seq_len(nrow(variance_df))) +
        labs(
            title = "Cumulative Explained Variance",
            x = "Number of Principal Components",
            y = "Cumulative Variance Explained (%)"
        ) +
        theme_bw()
}

#compute_model <- function(){

filepath <- c(
    "data/training_data/background/07072025_1.txt",
    "data/training_data/background/07072025_2.txt",
    "data/training_data/background/07072025_3.txt",
    "data/training_data/background/07072025_4.txt",
    "data/training_data/background/07072025_5.txt",
    "data/training_data/background/07072025_6.txt",
    #"data/training_data/monolayer/07012025_1.txt",
    #"data/training_data/monolayer/07022025_1.txt",
    #"data/training_data/monolayer/07022025_2.txt",
    #"data/training_data/monolayer/07022025_3.txt",
    #"data/training_data/monolayer/07022025_4.txt",
    #"data/training_data/monolayer/07022025_5.txt",
    #"data/training_data/monolayer/07022025_6.txt",
    #"data/training_data/monolayer/07022025_7.txt",
    #"data/training_data/monolayer/07022025_8.txt",
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
    #"data/training_data/bilayer/07022025_1.txt",
    "data/training_data/bilayer/07152025_1.txt",
    "data/training_data/bilayer/07152025_2.txt",
    "data/training_data/bilayer/07152025_3.txt",
    "data/training_data/bilayer/07152025_4.txt",
    "data/training_data/bilayer/08142025_1.txt",
    "data/training_data/bilayer/08142025_2.txt",
    "data/training_data/bilayer/08142025_3.txt"
    )

# PCA ON RAW SPECTRA

spectra <- lapply(filepath, fread)
intensity_matrix <- do.call(rbind, lapply(spectra, function(df){
    df <- df[df$V1 > 350 & df$V1 < 450, ]
    df$V2
    }))
intensity_matrix <- t(scale(t(intensity_matrix), center = TRUE, scale = TRUE))
raw_pca <- prcomp(intensity_matrix, center = TRUE, scale. = FALSE)
raw_scores <- data.frame(raw_pca$x)
labels <- as.factor(basename(dirname(filepath)))
raw_scores$Layer <- labels

ggplot(raw_scores, aes(PC1, PC4, colour = Layer)) +
    geom_point(size = 4)

# PCA ON ENGINEERED COMPONENTS

results <- bind_rows(lapply(seq_along(filepath), function(i) {
    process_spectrum(filepath[i], i)
}))

feature_vars <- results |>
    dplyr::select(
        diff_fit, fwhm1, fwhm2, 
        area_ratio, intensity_ratio, snr, rmse, r_squared
        )

feature_pca <- prcomp(feature_vars, center = TRUE, scale. = TRUE)
results <- results |>
    mutate(
        #PC1 = feature_pca$x[,1],
        #PC2 = feature_pca$x[,2],
        #PC3 = feature_pca$x[,3],
        #PC4 = feature_pca$x[,4],
        #PC5 = feature_pca$x[,5],
        PC1 = raw_scores$PC1,
        PC2 = raw_scores$PC2,
        PC3 = raw_scores$PC3,
        PC4 = raw_scores$PC4,
        PC5 = raw_scores$PC5
    )
results$Layer <- labels

ggplot(results, aes(PC4, PC5, colour = Layer)) +
    geom_point(size = 4) + 
    theme_bw()

feature_table <- results |>
    dplyr::select(
        Layer, 
        x_axis1, x_axis2,
        diff_peak,
        intensity_ratio,
        mu1, mu2,
        fwhm1, fwhm2, 
        A1, A2,
        area1, area2,
        area_ratio, 
        snr,
        rmse,
        r_squared,
        diff_fit,
        PC1,
        PC2,
        PC3,
        PC4,
        PC5
        )

scaled_features <- as.data.frame(scale(feature_table[,-c(1)]))
scaled_features$Layer <- feature_table$Layer
lda_model <- lda(
    Layer ~ 
        (
            x_axis1 + x_axis2 +
            diff_peak + 
            intensity_ratio +
            mu1 + mu2 +
            fwhm1 + fwhm2 + 
            A1 + A2 +
            area1 + area2 +
            area_ratio + 
            snr + 
            rmse + 
            r_squared + 
            diff_fit + 
            PC1 + 
            PC2 + 
            PC3 + 
            PC4 + 
            PC5
        ),
    data = scaled_features#,
    #CV = TRUE
)
#}

#lda_model <- compute_model()
pred <- lda_model
pred$class
table(Actual = feature_table$Layer, Predicted = pred$class)
mean(pred$class == feature_table$Layer)


#filepath <- paste0("data/training_data/monolayer/", "07102025_1", ".txt")
#data <- fread(filepath, header = FALSE)
#ggplot(data, aes(x = V1, y = V2)) +
#    geom_line() + 
#    theme_bw()

