rm(list = ls())

# 0. Libraries, Sources, Clustering
library(caret)
library(parallel)
source("R_Code/functions.R")

num_cores <- detectCores(logical = FALSE) - 1
cl <- makeCluster(num_cores, type = "PSOCK", outfile = "parallel_debug.log")
clusterEvalQ(cl, {
    library(minpack.lm) 
    library(tibble)
})
clusterExport(cl, c("gaussian", "double_gaussian"))

# 1. Load data ----
filepath <- list.files(
    path = "data/training_data", pattern = "\\.txt$", 
    recursive = TRUE, full.names = TRUE
)

filepath <- setdiff(
    filepath, 
    list.files(path = "data/training_data/bulk", pattern = "\\.txt$", full.names = TRUE)
    )

spectra <- lapply(filepath, fread)
results <- bind_rows(lapply(seq_along(filepath), function(i) {
    process_spectrum(filepath[i], i, cl)
}))

stopCluster(cl)

labels <- as.factor(basename(dirname(filepath)))
results$Layer <- labels
map_vector <- c(
    "background" = 0.00, "monolayer" = 0.70, 
    "bilayer" = 2.02, "bulk" = 4.00
    )
results$thickness <- map_vector[as.character(results$Layer)]

feature_table <- results |>
    dplyr::select(
        x_axis1, x_axis2, diff_peak, intensity_ratio, 
        mu1, mu2, fwhm1, fwhm2, A1, A2, area1, area2, area_ratio, snr, rmse,
        r_squared, diff_fit, Layer, thickness
    )

# 2. Create folds ----
set.seed(449)
folds <- createFolds(
    y = feature_table$Layer, k = 5,
    list = TRUE, returnTrain = FALSE
)

# 3. Cross Validate Function ----
cross_validate_model <- function(data, folds, selection){
    predictors  <- names(data)[selection]
    lda_pred    <- rep(NA, nrow(data))
    linear_pred <- rep(NA, nrow(data))
    ridge_pred  <- rep(NA, nrow(data))
    plsr_pred   <- rep(NA, nrow(data))
    svr_pred    <- rep(NA, nrow(data))
    rf_pred     <- rep(NA, nrow(data))
    nn_pred     <- rep(NA, nrow(data))

    for(i in seq_along(folds)){
        test_idx <- folds[[i]]
        train_idx <- setdiff(seq_len(nrow(data)), test_idx)
        
        train <- data[train_idx, ]
        test  <- data[test_idx, ]
        
        train_x <- train[, predictors, drop = FALSE]
        test_x  <- test[, predictors, drop = FALSE]
        
        center <- sapply(train_x, mean)
        scale  <- sapply(train_x, sd)
        
        scale[scale == 0] <- 1
        
        train_x <- as.data.frame(scale(train_x, center = center, scale = scale))
        test_x <- as.data.frame(scale(test_x, center = center, scale = scale))
        
        train_y <- train$thickness
        
        # LDA CLASSIFICATION
        fit <- lda(thickness ~ ., data = cbind(train_x, thickness = train_y))
        lda_class <- predict(fit, newdata = test_x)$class
        lda_pred[test_idx] <- c(
            background = 0, monolayer = 0.7, 
            bilayer = 1.4, bulk = 4.00
            )[lda_class]
        
        # LINEAR
        fit <- lm(thickness ~ ., data = cbind(train_x, thickness = train_y))
        linear_pred[test_idx] <- predict(fit, newdata = test_x)

        # RIDGE
        Xtrain <- as.matrix(train_x)
        Xtest  <- as.matrix(test_x)
        
        cv <- cv.glmnet(Xtrain, train_y, alpha = 0)
        fit <- glmnet(Xtrain, train_y, alpha = 0, lambda=cv$lambda.min)
        ridge_pred[test_idx] <- predict(fit, newx=Xtest)
        
        # PLSR
        fit <- plsr(thickness~., data=cbind(train_x, thickness=train_y), validation="CV")
        plsr_pred[test_idx] <- predict(fit, newdata=test_x, ncomp=fit$ncomp)
        
        # LINEAR
        fit <- lm(thickness ~ ., data = cbind(train_x, thickness = train_y))
        linear_pred[test_idx] <- predict(fit, newdata = test_x)
        
        # RIDGE
        Xtrain <- as.matrix(train_x)
        Xtest  <- as.matrix(test_x)
        
        cv <- cv.glmnet(Xtrain, train_y, alpha = 0)
        fit <- glmnet(Xtrain, train_y, alpha = 0, lambda=cv$lambda.min)
        ridge_pred[test_idx] <- predict(fit, newx=Xtest)
        
        # PLSR
        fit <- plsr(thickness~., data=cbind(train_x, thickness=train_y), validation="CV")
        opt_comp <- selectNcomp(fit, method = "onesigma", plot = FALSE) 
        if (opt_comp == 0) opt_comp <- 1 
        plsr_pred[test_idx] <- predict(fit, newdata=test_x, ncomp=opt_comp)
        
        # SVR
        fit <- svm(thickness~., data=cbind(train_x, thickness=train_y))
        svr_pred[test_idx] <- predict(fit, newdata=test_x)
        
        # RF
        fit <- randomForest(thickness~., data=cbind(train_x, thickness=train_y))
        rf_pred[test_idx] <- predict(fit, newdata=test_x)
        
        # NN
        fit <- nnet(
            thickness ~ ., data = cbind(train_x, thickness=train_y),
            size = 3, linout = TRUE, trace = FALSE, maxit = 1000
        )
        nn_pred[test_idx] <- predict(fit, newdata=test_x)
    }
    metrics <- rbind(
        cbind(Model = "LDA", model_metrics(data$thickness, lda_pred)), 
        cbind(Model = "Linear", model_metrics(data$thickness, linear_pred)),
        cbind(Model = "Ridge", model_metrics(data$thickness, ridge_pred)),
        cbind(Model = "PLSR", model_metrics(data$thickness, plsr_pred)),
        cbind(Model = "SVR", model_metrics(data$thickness, svr_pred)),
        cbind(Model = "RF", model_metrics(data$thickness, rf_pred)),
        cbind(Model = "NN", model_metrics(data$thickness, nn_pred))
    )
    
    metrics$Variables_Used <- paste(predictors, collapse=", ")
    metrics$Indices_Used <- list(selection)
    metrics$Num_Features <- length(selection)
    return(metrics)
}

# 4. Testing the model ----
num_cores <- detectCores(logical = FALSE) - 1
cl <- makeCluster(num_cores, type = "PSOCK", outfile = "parallel_debug.log")
clusterEvalQ(cl, {
    library(glmnet)
    library(pls)
    library(e1071)
    library(randomForest)
    library(nnet)
    library(MASS)
})
clusterExport(cl, c("feature_table", "folds", "cross_validate_model", "model_metrics"))

numbers <- 1:17
all_combos <- lapply(2:7, function(x) {combn(numbers, x, simplify = FALSE)})
all_combos <- unlist(all_combos, recursive = FALSE)

results_list <- pblapply(X = all_combos, FUN = function(selection){
    cross_validate_model(feature_table, folds, selection)
    }, cl = cl
)

stopCluster(cl)

final_results_df <- do.call(rbind, results_list)
rownames(final_results_df) <- NULL