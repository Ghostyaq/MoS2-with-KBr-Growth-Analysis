lm_p <- ggplot(large_area_features, aes(x = x, y = y, fill = thickness_lm)) +
    geom_tile() +
    coord_equal() +
    scale_fill_gradientn(
        colors = c("lightblue", "yellow", "orange", "red"),
        limits = c(-2, 3), 
        oob = squish, 
        breaks = c(-2, 0, 0.5, 1, 3),
        labels = c("≤ 0", "0", "0.5", "1", "3")
    ) +
    labs(fill = "Clustering", title = "Linear Model Predictions") + 
    theme_bw()

lda_p <- ggplot(large_area_features, aes(x = x, y = y, fill = thickness_lda)) +
    geom_tile() +
    coord_equal() +
    scale_fill_gradientn(colors = c("lightblue", "yellow", "red")) + 
    labs(fill = "Clustering", title = "Linear Discriminant Analysis Predictions") + 
    theme_bw()

rr_p <- ggplot(large_area_features, aes(x = x, y = y, fill = thickness_rr)) +
    geom_tile() +
    coord_equal() +
    scale_fill_gradientn(
        colors = c("purple", "blue", "orange", "red"),
        limits = c(-0.05, 2), 
        oob = squish, 
        breaks = c(-0.05, 0, 0.5, 1, 2),
        labels = c("≤ 0", "0", "0.5", "1", "2")
    ) +
    labs(fill = "Clustering", title = "Ridge Regression Predictions") + 
    theme_bw()

rf_p <- ggplot(large_area_features, aes(x = x, y = y, fill = thickness_rf)) +
    geom_tile() +
    coord_equal() +
    scale_fill_gradientn(
        colors = c("lightblue", "yellow", "red"),
        limits = c(0, 3), 
        oob = squish, 
        breaks = c(0, 2, 3),
        labels = c("≤ 0", "2", "3")
    ) +    labs(fill = "Clustering", title = "Random Forest Predictions") + 
    theme_bw()

vr_p <- ggplot(large_area_features, aes(x = x, y = y, fill = thickness_vr)) +
    geom_tile() +
    coord_equal() +
    scale_fill_gradientn(colors = c("lightblue", "yellow", "red")) + 
    labs(fill = "Clustering", title = "Support Vector Predictions") + 
    theme_bw()

pl_p <- ggplot(large_area_features, aes(x = x, y = y, fill = thickness_pl)) +
    geom_tile() +
    coord_equal() +
    scale_fill_gradientn(
        colors = c("purple", "blue", "orange", "red"),
        limits = c(-0.05, 2), 
        oob = squish, 
        breaks = c(-0.05, 0, 0.5, 1, 2),
        labels = c("≤ 0", "0", "0.5", "1", "2")
    ) +
    labs(fill = "Clustering", title = "Partial Least Squares Regression Predictions") + 
    theme_bw()

nn_p <- ggplot(large_area_features, aes(x = x, y = y, fill = thickness_nn)) +
    geom_tile() +
    coord_equal() +
    scale_fill_gradientn(
        colors = c("lightblue", "yellow", "red"),
        limits = c(0, 3), 
        oob = squish, 
        breaks = c(0, 2, 3),
        labels = c("≤ 0", "2", "3")
    ) +
    labs(fill = "Clustering", title = "Artificial Neural Net Predictions") + 
    theme_bw()

p <- ggplot(large_area_features, aes(x = x, y = y, fill = mean_thickness)) +
    geom_tile() +
    coord_equal() +
    scale_fill_gradientn(
        colors = c("purple", "blue", "orange", "red"),
        limits = c(-0.05, 2), 
        oob = squish, 
        breaks = c(-0.05, 0, 0.5, 1, 2),
        labels = c("≤ 0", "0", "0.5", "1", "2")
    ) +
    labs(fill = "Clustering", , title = "Mean Thickness of All Predictions") + 
    theme_bw()