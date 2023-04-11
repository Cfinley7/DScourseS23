#PS9
#Caden Finley

library(tidymodels)
library(glmnet)
library(magrittr)
library(readr)
set.seed(123456)
housing <- read_table("http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", col_names = FALSE)
names(housing) <- c("crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat","medv")

housing_split <- initial_split(housing, prop = 0.8)

housing_train <- training(housing_split)
housing_test  <- testing(housing_split)

housing_recipe <- recipe(medv ~ ., data = housing) %>% 
  # convert outcome variable to logs
  step_log(all_outcomes()) %>%
  # convert 0/1 chas to a factor
  step_bin2factor(chas) %>%
  # create interaction term between crime and nox 
  step_interact(terms = ~ crim:zn:indus:rm:age:rad:tax:
  ptratio:b:lstat:dis:nox) %>%
  # create square terms of some continuous variables
  step_poly(crim,zn,indus,rm,age,rad,tax,ptratio,b, lstat,dis,nox, degree=6)

# run the recipe
housing_prep <- housing_recipe %>% prep(housing_train , retain = TRUE)
housing_train_prepped <- housing_prep %>% juice
housing_test_prepped <- housing_prep %>% bake(new_data = housing_test)

# create x and y training and test data
housing_train_x <- housing_train_prepped %>% select(-medv) 
housing_test_x <- housing_test_prepped %>% select(-medv)
housing_train_y <- housing_train_prepped %>% select( medv) 
housing_test_y <- housing_test_prepped %>% select( medv)

# dimension of housing_train data is 404 rows and 14 columns
# have 1 more x variable than the original housing data

# LASSO model
# log median house value

tune_spec <- linear_reg(
  penalty = tune(), # tuning parameter
  mixture = 1       # 1 = lasso, 0 = ridge
) %>% 
  set_engine("glmnet") %>%
  set_mode("regression")
# define a grid over which to try different values of lambda
lambda_grid <- grid_regular(penalty(), levels = 50)
# 10-fold cross-validation
rec_folds <- vfold_cv(housing_train_prepped, v = 6)
optimal_lambda_lasso <- tune_spec$bestTune$lambda
cat("Optimal value of lambda for LASSO:", optimal_lambda_lasso, "\n")

# ridge regression model 
library(caret)
ridge_model <- train(log_median_house_value ~ ., data = train_data, method = "glmnet",
                     trControl = trainControl(method = "cv", number = 6),
                     tuneGrid = expand.grid(alpha = 0, lambda = 10^seq(-5, 5, by = 0.1)))

# Optimal value of lambda for Ridge
optimal_lambda_ridge <- ridge_model$bestTune$lambda
cat("Optimal value of lambda for Ridge:", optimal_lambda_ridge, "\n")