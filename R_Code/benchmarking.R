rm(list = ls())
load("data/RData/base_analysis.RData")

library(MASS)
library(pls)
library(e1071)
library(pbapply)
library(parallel)
library(randomForest)
library(nnet)
library(glmnet)
library(tidyverse)

model_metrics <- function(actual, predicted) {
    rmse <- sqrt(mean((actual - predicted) ^ 2))
    mae <- mean(abs(actual - predicted))
    r2 <- 1 - sum((actual - predicted) ^ 2) / sum((actual - mean(actual)) ^ 2)
    data.frame(RMSE = rmse, MAE = mae, R2 = r2)
}

real_measurements <- scaled_features$thickness
temp <- scaled_features |> dplyr::select(!c(Layer, thickness))

numbers <- 1:17
all_combos <- lapply(2:length(numbers), function(x) {
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
    library(randomForest)
    library(nnet)
    library(glmnet)
})

thickness <- scaled_features$thickness
clusterExport(cl, c("temp", "thickness", "model_metrics"))

calculate_metrics <- function(selection, data, thickness, Layer) {
    selected_vars <- names(data)[selection]
    vars_used_string <- paste(selected_vars, collapse = ", ")
    subset_data <- data[, selection, drop = FALSE]

    # -------- LDA --------
    lda_metrics <- tryCatch ({
        lda_model <- lda(Layer ~ ., data = cbind(Layer, subset_data), prior = c(1/3, 1/3, 1/3))
        lda_prediction_class <- predict(lda_model, newdata = subset_data)$class
        lda_prediction <- (c(
            background = 0,
            monolayer = 0.7,
            bilayer = 2.02
        )[lda_prediction_class])
        model_metrics(thickness, lda_prediction)
    }, error = function(e) {
        print(e$message)
        data.frame(RMSE = e$message, MAE = NA, R2 = NA)
        })
    
    subset_data$thickness <- thickness
    # -------- PLSR (dynamic ncomp optimization) --------
    plsr_model <- plsr(thickness ~ ., data = subset_data, scale = FALSE)
    max_comp <- plsr_model$ncomp
    rmse <- sapply(1:max_comp, function(i){
            pred <- as.vector(predict(plsr_model, ncomp = i))
            sqrt(mean((subset_data$thickness - pred) ^ 2))
        }
    )
    best_ncomp <- which.min(rmse)
    plsr_prediction <- as.vector(predict(plsr_model, ncomp = best_ncomp))
    
    # -------- LINEAR --------
    linear_model <- lm(thickness ~ ., data = subset_data)
    linear_prediction <- predict(linear_model)
    
    # -------- SVR --------
    svr_model <- svm(
        thickness ~ ., data = subset_data,
        type = "eps-regression", kernel = "radial"
        )
    svr_prediction <- predict(svr_model)
    
    # -------- RR --------
    X <- as.matrix(dplyr::select(subset_data, !thickness))
    ridge_cv_model <- cv.glmnet(X, subset_data$thickness, alpha = 0)
    best_lambda <- ridge_cv_model$lambda.min
    ridge_model <- glmnet(X, thickness, alpha = 0, lambda = best_lambda)
    ridge_prediction <- predict(ridge_model, s = best_lambda, newx = X)
    
    # -------- RF -------- 
    forest_model <- randomForest(
        thickness ~ ., data = subset_data, 
        ntree = 500, mtry = 2, importance = TRUE
    )
    forest_prediction <- predict(forest_model)
    
    # -------- Neural Network --------
    nn_model <- nnet(
        thickness ~ ., data = subset_data, size = 3, linout = TRUE, 
        decay = 0.01, maxit = 1000, trace = FALSE
    )
    neural_prediction <- predict(nn_model)
    
    # -------- Metrics Calculation --------
    plsr_metrics <- model_metrics(thickness, plsr_prediction)
    linear_metrics <- model_metrics(thickness, linear_prediction)
    svr_metrics <- model_metrics(thickness, svr_prediction)
    rf_metrics <- model_metrics(thickness, forest_prediction)
    nn_metrics <- model_metrics(thickness, neural_prediction)
    rr_metrics <- model_metrics(thickness, ridge_prediction)
    
    # Combine results and include the chosen PLSR component count
    benchmarking_results <- rbind(
        cbind(Model = "LDA", lda_metrics, PLSR_Components = NA),
        cbind(Model = "Lin Reg", linear_metrics, PLSR_Components = NA),
        cbind(Model = "PLSR", plsr_metrics, PLSR_Components = best_ncomp),
        cbind(Model = "SVR", svr_metrics, PLSR_Components = NA),
        cbind(Model = "RF", rf_metrics, PLSR_Components = NA),
        cbind(Model = "NN", nn_metrics, PLSR_Components = NA),
        cbind(Model = "RR", rr_metrics, PLSR_Components = NA)
    )
    
    benchmarking_results$Num_Features <- length(selection)
    benchmarking_results$Indices_Used <- list(selection)
    benchmarking_results$Variables_Used <- vars_used_string
    return(benchmarking_results)
}

# 3. Run in Parallel with a Progress Bar
results_list <- pblapply(
    X = all_combos, 
    FUN = calculate_metrics, 
    data = temp, 
    thickness = thickness, 
    Layer = scaled_features$Layer,
    cl = cl
)

stopCluster(cl)
final_results_df <- do.call(rbind, results_list)
rownames(final_results_df) <- NULL

# --------------------------- #

#load("data/Rdata/benchmark_non_cv.RData")

#LDA
lda_best <- filter(final_results_df, Model == "LDA") |> arrange(desc(R2)) |> head(20)
lin_reg_best <- filter(final_results_df, Model == "Lin Reg") |> arrange(desc(R2)) |> head(20)
plsr_best <- filter(final_results_df, Model == "PLSR") |> arrange(desc(R2)) |> head(20)
svr_best <- filter(final_results_df, Model == "SVR") |> arrange(desc(R2)) |> head(20)
rf_best <- filter(final_results_df, Model == "RF") |> arrange(desc(R2)) |> head(20)
nn_best <- filter(final_results_df, Model == "NN") |> arrange(desc(R2)) |> head(20)
rr_best <- filter(final_results_df, Model == "RR") |> arrange(desc(R2)) |> head(20)

best <- rbind(lda_best, lin_reg_best, plsr_best, svr_best, rf_best, nn_best, rr_best)
interested_indices <- best$Indices_Used

