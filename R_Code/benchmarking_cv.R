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

loocv <- function(train_fun, predict_fun, data){
    n <- nrow(data)
    prediction <- numeric(n)
    for(i in 1:n){
        train <- data[-i, ]
        test  <- data[i, , drop = FALSE]
        model <- train_fun(train)
        prediction[i] <- predict_fun(model, test)
    }
    prediction
}

real_measurements <- scaled_features$thickness
temp <- scaled_features |> select(!c(Layer, thickness))

#numbers <- 2:17
#all_combos <- lapply(1:length(numbers), function(x) {
#    combn(numbers, x, simplify = FALSE)
#})
#all_combos <- unlist(all_combos, recursive = FALSE)

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
clusterExport(cl, c("temp", "thickness", "model_metrics", "loocv"))

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
    linear_prediction <- loocv(
        train_fun = function(train) {lm(thickness ~ ., data = train)},
        predict_fun = function(model, test) {predict(model, test)},
        data = subset_data
    )
    
    # -------- SVR --------
    svr_prediction <- loocv(
        train_fun = function(train){
            svm(thickness ~ ., data = train, kernel = "radial")
        },
        predict_fun = function(model, test){predict(model, test)},
        data = subset_data
    )
    
    # -------- RR --------
    ridge_prediction <- loocv(
        train_fun = function(train){
            X_train <- as.matrix(dplyr::select(train, -thickness))
            y_train <- train$thickness
            model <- cv.glmnet(X_train, y_train, alpha = 0, nfolds = nrow(train))
        },
        predict_fun = function(model, test){
            X_test <- as.matrix(dplyr::select(test, -thickness))
            predict(model, newx = X_test, s = "lambda.min")
            },
        data = subset_data
    )
    
    # -------- RF -------- 
    forest_prediction <- loocv(
        train_fun = function(train){
            forest_model <- randomForest(
                thickness ~ ., data = train,  ntree = 500, mtry = 2
            )
        },
        predict_fun = function(model, test){predict(model, newdata = test)},
        data = subset_data
    )
    
    # -------- Neural Network --------
    neural_prediction <- loocv(
        train_fun = function(train){
            model <- nnet(
                thickness ~ ., data = train, size = 3,
                linout = TRUE, decay = 0.01, trace = FALSE
            )
        },
        predict_fun = function(model, test){predict(model, newdata = test)},
        data = subset_data
    )
    
    # -------- Metrics Calculation --------
    lda_metrics <- model_metrics(real_measurements, lda_prediction)
    plsr_metrics <- model_metrics(real_measurements, plsr_prediction)
    linear_metrics <- model_metrics(real_measurements, linear_prediction)
    svr_metrics <- model_metrics(real_measurements, svr_prediction)
    rf_metrics <- model_metrics(real_measurements, forest_prediction)
    nn_metrics <- model_metrics(real_measurements, neural_prediction)
    rr_metrics <- model_metrics(real_measurements, ridge_prediction)
    
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
    real_measurements = thickness,
    cl = cl
)

stopCluster(cl)
final_results_df <- do.call(rbind, results_list)
rownames(final_results_df) <- NULL
