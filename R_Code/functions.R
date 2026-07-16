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