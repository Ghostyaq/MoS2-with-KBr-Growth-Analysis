rm(list = ls())
library(tidyverse)
library(minpack.lm)

source("R_Code/functions.R")

raw <- fread("data/default LAS/300x300/300x300.txt")
data <- normalize_data(raw)

cl <- makeCluster(1, type = "PSOCK")
peak_summary <- find_peak_locations(data, cl)
stopCluster(cl)

plot_gaussian_fit <- function(raw, peak_locations, spectrum_id) {
    data <- raw[raw$V1 > 375 & raw$V1 < 420, ]
    x <- data$V1
    y <- data[[spectrum_id + 1]]
    
    peak <- peak_locations[peak_locations$id == spectrum_id, ]
    
    fit <- nlsLM(
        y ~ double_gaussian(
            x, A1, mu1, sigma1,
            A2, mu2, sigma2, C
        ),
        start = list(
            A1 = peak$intensity1,
            mu1 = peak$x_axis1,
            sigma1 = 3,
            A2 = peak$intensity2,
            mu2 = peak$x_axis2,
            sigma2 = 3,
            C = min(y)
        ),
        lower = c(0, 370, 0.5, 0, 390, 0.5, 0),
        upper = c(1100, 400, 20, 1100, 430, 20, 1100)
    )
    
    # Extract fitted parameters
    p <- coef(fit)
    
    # Generate smooth fitted Gaussian
    x_fit <- seq(min(x), max(x), length.out = 1000)
    
    y_fit <- double_gaussian(
        x_fit,
        p["A1"], p["mu1"], p["sigma1"],
        p["A2"], p["mu2"], p["sigma2"],
        p["C"]
    )
    
    plot(
        x, y,
        type = "l",
        xlab = "Raman Shift (cm^-1)",
        ylab = "Intensity (a.u.)",
        ylim = c(253, 275), 
        main = paste("Double Gaussian Fit (Background)")
    )
    
    lines(
        x_fit,
        y_fit,
        lwd = 3, 
        col = "#a7000a"
    )
}

plot_gaussian_fit(data, peak_summary, spectrum_id = 5)
