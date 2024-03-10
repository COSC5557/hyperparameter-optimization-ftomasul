# Finn Tomasula Martin
# COSC-4557
# Hyperparameter Optimization
# This file contains the code for the hyperparameter optimization exercise

# Clear environment
rm(list = ls())
while (!is.null(dev.list())) dev.off()

# Load libraries
library(caret)
library(randomForest)
library(e1071)
library(rBayesianOptimization)

# Load in data
wine <- read.csv("winequality-red.csv", sep=";")

# Add new new column to classify wine as good or bad based on a cutoff
cutoff <- mean(wine$quality)
wine$quality.bin <- ifelse(wine$quality >= cutoff, "good", "bad")
wine$quality.bin <- factor(wine$quality.bin)

# Split data into 10 folds
set.seed(123)
num_folds <- 10
folds <- createFolds(wine$quality.bin, k = num_folds, list = TRUE, returnTrain = TRUE)

## RANDOM FOREST ##

# Define objective function for random forest algorithm
rand_for_function <- function(num_trees, num_vars) {
  num_trees <- as.integer(num_trees)
  num_vars <- as.integer(num_vars)
  
  accuracy_list <- numeric(num_folds)
  
  for(i in 1:num_folds) {
    indices <- folds[[i]]
    train <- wine[indices, ]
    test <- wine[-indices, ]
    model <- randomForest(quality.bin ~.-quality, data = train, ntree = num_trees, mtry = num_vars)
    preds <- predict(model, newdata = test)
    len <- length(test$quality.bin)
    correct <- sum(preds == test$quality.bin)
    accuracy_list[i] <- correct / len
  }
  
  mean_accuracy = mean(accuracy_list)
  
  return(list(Score = mean_accuracy, Pred = 0))
}

# Define bounds for the parameters
rand_for_bounds <- list(
  num_trees = c(50, 200),
  num_vars = c(1, 11)
)

# Run Bayesian optimization on random forest with respect to number of trees and number of variables 
rand_for_optimized <- BayesianOptimization(
  rand_for_function,
  rand_for_bounds,
  init_points = 5,
  n_iter = 20,
  acq = "poi",
  eps = 0.1,
  verbose = TRUE
)

# Visualize the results of the random forest optimization process
num_trees <- rand_for_optimized$History$num_trees
num_vars <- rand_for_optimized$History$num_vars
accuracy <- rand_for_optimized$History$Value

colors <- colorRampPalette(c("lightblue", "darkblue"))(length(unique(accuracy)))

point_colors <- colors[cut(accuracy, breaks = length(colors), include.lowest = TRUE)]

rand_for_plot <- plot(x = num_trees, y = num_vars, xlim = c(50, 200), ylim = c(1, 11),
                      pch = 16, col = point_colors,
                      xlab = "Number of Trees", ylab = "Number of Variables", 
                      main = "Random Forest Optimization")

## SVM ##

# Define objective function for SVM
svm_function <- function(gam, C) {
  accuracy_list <- numeric(num_folds)
  
  for(i in 1:num_folds) {
    indices <- folds[[i]]
    train <- wine[indices, ]
    test <- wine[-indices, ]
    model <- svm(quality.bin ~.-quality, data = train, gamma = gam, cost = C)
    preds <- predict(model, newdata = test)
    len <- length(test$quality.bin)
    correct <- sum(preds == test$quality.bin)
    accuracy_list[i] <- correct / len
  }
  
  mean_accuracy = mean(accuracy_list)
  
  return(list(Score = mean_accuracy, Pred = 0))
}

# Define bounds for the parameters
svm_bounds <- list(
  gam = c(0.1, 10),
  C = c(0.01, 1)
)

# Run Bayesian optimization on SVM with respect to gamma and cost
svm_optimized <- BayesianOptimization(
  svm_function,
  svm_bounds,
  init_points = 5,
  n_iter = 20,
  acq = "poi",
  eps = 0.1,
  verbose = TRUE
)

# Visualize the results of the random forest optimization process
gam <- svm_optimized$History$gam
C <- svm_optimized$History$C
accuracy <- svm_optimized$History$Value

colors <- colorRampPalette(c("lightblue", "darkblue"))(length(unique(accuracy)))

point_colors <- colors[cut(accuracy, breaks = length(colors), include.lowest = TRUE)]

rand_for_plot <- plot(x = gam, y = C, xlim = c(0.1, 10), ylim = c(0.01, 1),
                      pch = 16, col = point_colors,
                      xlab = "Gamma", ylab = "Cost", 
                      main = "SVM Optimization")

