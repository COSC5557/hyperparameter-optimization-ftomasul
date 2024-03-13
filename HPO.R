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

# Split data into train/test set for outer evaluation
set.seed(123)
wine_ind <- createDataPartition(wine$quality.bin, p = 0.8, list = FALSE)
wine_train <- wine[wine_ind, ]
wine_test <- wine[-wine_ind, ]
test_len <- length(wine_test$quality.bin)

# Split train data into 10 folds for inner evaluation
set.seed(123)
num_folds <- 10
folds <- createFolds(wine_train$quality.bin, k = num_folds, list = TRUE, returnTrain = TRUE)

## RANDOM FOREST ##

# Results of random forest model without any tuning
rf_base_model <- randomForest(quality.bin ~.-quality, data = wine_train)
rf_base_preds <- predict(rf_base_model, newdata = wine_test)
rf_base_correct <- sum(rf_base_preds == wine_test$quality.bin)
rf_base_accuracy <- rf_base_correct / test_len

# Define objective function for random forest algorithm
rand_for_function <- function(num_trees, num_vars, node_size) {
  num_trees <- as.integer(num_trees)
  num_vars <- as.integer(num_vars)
  node_size <- as.integer(node_size)
  
  accuracy_list <- numeric(num_folds)
  
  for(i in 1:num_folds) {
    indices <- folds[[i]]
    train <- wine_train[indices, ]
    test <- wine_train[-indices, ]
    model <- randomForest(quality.bin ~.-quality, data = train, ntree = num_trees, mtry = num_vars, nodesize = node_size)
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
  num_trees = c(50L, 500L),
  num_vars = c(1L, 11L),
  node_size = c(1L, 10L)
)

# Run Bayesian optimization on random forest with respect to number of trees and number of variables 
rand_for_optimized <- BayesianOptimization(
  rand_for_function,
  rand_for_bounds,
  init_points = 5,
  n_iter = 100,
  acq = "ucb",
  eps = 0.01,
  verbose = TRUE
)

# Evaluate random forest model with optimized parameters
rf_opti_par <- rand_for_optimized$Best_Par
opti_num_trees <- as.integer(rf_opti_par[1])
opti_num_vars <- as.integer(rf_opti_par[2])
opti_node_size <- as.integer(rf_opti_par[3])
rf_opti_model <- randomForest(quality.bin ~.-quality, data = wine_train, ntree = opti_num_trees, mtry = opti_num_vars, nodesize = opti_node_size)
rf_opti_preds <- predict(rf_opti_model, newdata = wine_test)
rf_opti_correct <- sum(rf_opti_preds == wine_test$quality.bin)
rf_opti_accuracy <- rf_opti_correct / test_len

## SVM ##

# Results of SVM model without any tuning
svm_base_model <- svm(quality.bin ~.-quality, data = wine_train)
svm_base_preds <- predict(svm_base_model, newdata = wine_test)
svm_base_correct <- sum(svm_base_preds == wine_test$quality.bin)
svm_base_accuracy <- svm_base_correct / test_len

# Define objective function for SVM
svm_function <- function(ker, gam, C) {
  kern <- "radial"
  if(ker == 1) {
    kern <- "linear"
  } else if(ker == 2) {
    kern <- "polynomial"
  } else if(ker == 3) {
    kern <- "radial"
  } else if(ker == 4) {
    kern <- "sigmoid"
  }
  
  accuracy_list <- numeric(num_folds)
  
  for(i in 1:num_folds) {
    indices <- folds[[i]]
    train <- wine_train[indices, ]
    test <- wine_train[-indices, ]
    model <- svm(quality.bin ~.-quality, data = train, kernel = kern, gamma = gam, cost = C)
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
  ker = c(1L, 4L),
  gam = c(0.1, 1),
  C = c(0.1, 1)
)

# Run Bayesian optimization on SVM with respect to gamma and cost
svm_optimized <- BayesianOptimization(
  svm_function,
  svm_bounds,
  init_points = 5,
  n_iter = 100,
  acq = "poi",
  eps = 0.01,
  verbose = TRUE
)

# Evaluate SVM model with optimized parameters
svm_opti_par <- svm_optimized$Best_Par
temp <- as.integer(svm_opti_par[1])
if(temp == 1) {
  opti_kern <- "linear"
} else if(temp == 2) {
  opti_kern <- "polynomial"
} else if(temp == 3) {
  opti_kern <- "radial"
} else if(temp == 4) {
  opti_kern <- "sigmoid"
}
opti_gamma <- as.numeric(svm_opti_par[2])
opti_cost <- as.numeric(svm_opti_par[3])
svm_opti_model <- svm(quality.bin ~.-quality, data = wine_train, kernel = opti_kern, gamma = opti_gamma, cost = opti_cost)
svm_opti_preds <- predict(svm_opti_model, newdata = wine_test)
svm_opti_correct <- sum(svm_opti_preds == wine_test$quality.bin)
svm_opti_accuracy <- svm_opti_correct / test_len

## Results ##
rf_opti_par
rf_base_accuracy
rf_opti_accuracy
svm_opti_par
svm_base_accuracy
svm_opti_accuracy


