rm(list = ls())
load("data/RData/base_analysis.RData")

library(MASS)
library(pls)
library(e1071)
library(pbapply)
library(parallel)

model_metrics <- function(actual, predicted) {
    rmse <- sqrt(mean((actual - predicted) ^ 2))
    mae <- mean(abs(actual - predicted))
    r2 <- 1 - sum((actual - predicted) ^ 2) / sum((actual - mean(actual)) ^ 2)
    data.frame(RMSE = rmse, MAE = mae, R2 = r2)
}

real_measurements <- scaled_features$thickness
temp <- scaled_features |> select(!c(Layer, thickness))

numbers <- 1:17
all_combos <- lapply(1:length(numbers), function(x) {
    combn(numbers, x, simplify = FALSE)
})
all_combos <- unlist(all_combos, recursive = FALSE)

# 1. Setup the Parallel Cluster
num_cores <- max(1, detectCores() - 1)
cl <- makeCluster(num_cores)

clusterEvalQ(cl, {
    library(MASS)
    library(pls)
    library(e1071)
})

thickness <- scaled_features$thickness
clusterExport(cl, c("temp", "thickness", "model_metrics"))

calculate_metrics <- function(selection, data, thickness, real_measurements) {
    selected_vars <- names(data)[selection]
    vars_used_string <- paste(selected_vars, collapse = ", ")
    subset_data <- data[, selection, drop = FALSE]
    subset_data$thickness <- thickness
    
    # -------- LDA --------
    lda_model_benchmark <- lda(thickness ~ ., data = subset_data, CV = TRUE)
    lda_prediction <- c(
        "background" = 0, 
        "monolayer" = 0.7, 
        "bilayer" = 1.4
        )[lda_model_benchmark$class]
    
    # -------- PLSR (dynamic ncomp optimization) --------
    plsr_model <- plsr(
        thickness ~ ., data = subset_data, 
        validation = "LOO", scale = FALSE
        )
    # Extract MSEP cross-validation values (excluding the 0-component intercept model)
    cv_errors <- RMSEP(plsr_model, estimate = "CV")$val[1, 1, -1]
    best_ncomp <- which.min(cv_errors)
    plsr_prediction <- as.vector(plsr_model$validation$pred[, , best_ncomp])
    
    # -------- LINEAR --------
    linear_model <- lm(thickness ~ ., data = subset_data)
    linear_prediction <- predict(linear_model)
    
    # -------- SVR --------
    svr_model <- svm(
        thickness ~ ., data = subset_data,
        type = "eps-regression", kernel = "radial"
        )
    svr_prediction <- predict(svr_model)
    
    # -------- Metrics Calculation --------
    lda_metrics <- model_metrics(real_measurements, lda_prediction)
    plsr_metrics <- model_metrics(real_measurements, plsr_prediction)
    linear_metrics <- model_metrics(real_measurements, linear_prediction)
    svr_metrics <- model_metrics(real_measurements, svr_prediction)
    
    # Combine results and include the chosen PLSR component count
    benchmarking_results <- rbind(
        cbind(Model = "LDA", lda_metrics, PLSR_Components = NA),
        cbind(Model = "Lin Reg", linear_metrics, PLSR_Components = NA),
        cbind(Model = "PLSR", plsr_metrics, PLSR_Components = best_ncomp),
        cbind(Model = "SVR", svr_metrics, PLSR_Components = NA)
    )
    
    benchmarking_results$Variables_Used <- vars_used_string
    return(benchmarking_results)
}

# 3. Run in Parallel with a Progress Bar
results_list <- pblapply(
    X = all_combos, 
    FUN = calculate_metrics, 
    data = temp, 
    thickness = thickness, 
    real_measurements = thickness,
    cl = cl
)

stopCluster(cl)
final_results_df <- do.call(rbind, results_list)