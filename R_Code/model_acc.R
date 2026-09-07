rm(list = ls())
source("R_Code/functions.R")
load("data/RData/old_data_features.RData")

### LINEAR DISCRIMINATORY ANALYSIS ###
lda_model_benchmark <- lda(
    thickness ~ intensity_ratio + mu2 + area1,
    data = scaled_features, CV = TRUE
)

table(Actual = scaled_features$thickness, Predicted = lda_model_benchmark$class)
lda_model <- lda(
    Layer ~ intensity_ratio + mu2 + area1,
    data = scaled_features
)

### LINEAR REGRESSION ###
linear_model <- lm(
    thickness ~ x_axis1 + A2 + area1 + rmse,
    data = scaled_features)

### RIDGE REGRESSION ###
X <- as.matrix(scaled_features[c(11, 13, 15)])
y <- scaled_features$thickness
ridge_cv_model <- cv.glmnet(X, y, alpha = 0)
best_lambda <- ridge_cv_model$lambda.min
print(best_lambda)
plot(ridge_cv_model)
ridge_model <- glmnet(X, y, alpha = 0, lambda = best_lambda)
coef(ridge_model)

### RANDOM FOREST REGRESSION ###
forest_model <- randomForest(
    thickness ~ (x_axis1 + mu1 + fwhm2 + area_ratio + r_squared),
    data = scaled_features, ntree = 500, mtry = 2, importance = TRUE
)

### SUPPORT VECTOR REGRESSION ###
svr_model <- svm(
    thickness ~ x_axis1 + intensity_ratio + fwhm1 + fwhm2 + area_ratio + snr,
    data = scaled_features, type = "eps-regression", kernel = "radial"
)

###### PARTIAL LINEAR REGRESSION ######
plsr_model <- plsr(
    thickness ~ intensity_ratio + mu1 + fwhm1 + A1 + A2 + area_ratio + rmse,
    data = scaled_features,
    validation = "LOO",
    scale = FALSE
)
validationplot(plsr_model, val.type = "RMSEP")

############# NEURAL?! #############
nn_model <- nnet(
    thickness ~ (mu1 + fwhm1 + A1 + area_ratio + rmse),
    data = scaled_features, 
    size = 3,      # hidden neurons
    linout = TRUE, # regression instead of classification
    decay = 0.01,  # weight decay to reduce overfitting
    maxit = 1000,
    trace = TRUE
)

temp <- scaled_features

load("data/RData/new_data.RData")
original <- scaled_features
scaled_features <- as.matrix(scaled_features[, -c(18, 19)])

lda_pred <- predict(lda_model, newdata = as.data.frame(scaled_features[, c(4, 6, 11)]))
lm_pred <- predict(linear_model, newdata = as.data.frame(scaled_features[, c(1, 10, 11, 15)]))
rr_pred <- predict(ridge_model, s = best_lambda, newx = scaled_features[, c(11, 13, 15)])
rf_pred <- predict(forest_model, newdata = as.data.frame(scaled_features[, c(1, 5, 8, 13, 16)]))
vr_pred <- predict(svr_model, newdata = as.data.frame(scaled_features[, c(1, 4, 7, 8, 13, 14)]))
pl_pred <- predict(
    plsr_model,  ncomp = 4,
    newdata = as.data.frame(scaled_features[, c(4, 5, 7, 9, 10, 13, 15)])
)
nn_pred <- predict(nn_model, newdata = as.data.frame(scaled_features[, c(5, 7, 9, 13, 15)]))

comparisons <- data.frame(
    lda_pred$class, lm_pred, rr_pred, rf_pred, vr_pred, 
    pl_pred, nn_pred, thickness = original$thickness
    )

final_results <- lapply(
    list(lm_pred, rr_pred, rf_pred, vr_pred, pl_pred, nn_pred), 
    model_metrics,
    predicted = original$thickness
    ) |> 
    list_rbind()
final_results$model <- c("lm", "rr", "rf", "vr", "pl", "nn")
