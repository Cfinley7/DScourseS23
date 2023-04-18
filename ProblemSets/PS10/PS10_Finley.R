#PS10 Finley

library(tidyverse)
library(tidymodels)
library(magrittr)
library(modelsummary)
library(rpart)
library(e1071)
library(kknn)
library(nnet)
library(kernlab)
library(readr)

set.seed(100)

income <- read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", col_names = FALSE)
names(income) <- c("age","workclass","fnlwgt","education","education.num","marital.status","occupation","relationship","race","sex","capital.gain","capital.loss","hours","native.country","high.earner")

# Clean up the data
######################
# Drop unnecessary columns
income %<>% select(-native.country, -fnlwgt, education.num)
# Make sure continuous variables are formatted as numeric
income %<>% mutate(across(c(age,hours,education.num,capital.gain,capital.loss), as.numeric))
# Make sure discrete variables are formatted as factors
income %<>% mutate(across(c(high.earner,education,marital.status,race,workclass,occupation,relationship,sex), as.factor))
# Combine levels of factor variables that currently have too many levels
income %<>% mutate(education = fct_collapse(education,
                                            Advanced    = c("Masters","Doctorate","Prof-school"), 
                                            Bachelors   = c("Bachelors"), 
                                            SomeCollege = c("Some-college","Assoc-acdm","Assoc-voc"),
                                            HSgrad      = c("HS-grad","12th"),
                                            HSdrop      = c("11th","9th","7th-8th","1st-4th","10th","5th-6th","Preschool") 
),
marital.status = fct_collapse(marital.status,
                              Married      = c("Married-civ-spouse","Married-spouse-absent","Married-AF-spouse"), 
                              Divorced     = c("Divorced","Separated"), 
                              Widowed      = c("Widowed"), 
                              NeverMarried = c("Never-married")
), 
race = fct_collapse(race,
                    White = c("White"), 
                    Black = c("Black"), 
                    Asian = c("Asian-Pac-Islander"), 
                    Other = c("Other","Amer-Indian-Eskimo")
), 
workclass = fct_collapse(workclass,
                         Private = c("Private"), 
                         SelfEmp = c("Self-emp-not-inc","Self-emp-inc"), 
                         Gov     = c("Federal-gov","Local-gov","State-gov"), 
                         Other   = c("Without-pay","Never-worked","?")
), 
occupation = fct_collapse(occupation,
                          BlueCollar  = c("?","Craft-repair","Farming-fishing","Handlers-cleaners","Machine-op-inspct","Transport-moving"), 
                          WhiteCollar = c("Adm-clerical","Exec-managerial","Prof-specialty","Sales","Tech-support"), 
                          Services    = c("Armed-Forces","Other-service","Priv-house-serv","Protective-serv")
)
)


######################
# tidymodels time!
######################
income_split <- initial_split(income, prop = 0.8)
income_train <- training(income_split)
income_test  <- testing(income_split)

#####################
# logistic regression
#####################
print('Starting LOGIT')
# set up the task and the engine
tune_logit_spec <- logistic_reg(
  penalty = tune(), # tuning parameter
  mixture = 1       # 1 = lasso, 0 = ridge
) %>% 
  set_engine("glmnet") %>%
  set_mode("classification")

# define a grid over which to try different values of the regularization parameter lambda
lambda_grid <- grid_regular(penalty(), levels = 50)

# 3-fold cross-validation
rec_folds <- vfold_cv(income_train, v = 3)

# Workflow
rec_wf <- workflow() %>%
  add_model(tune_logit_spec) %>%
  add_formula(high.earner ~ education + marital.status + race + workclass + occupation + relationship + sex + age + capital.gain + capital.loss + hours)

# Tuning results
rec_res <- rec_wf %>%
  tune_grid(
    resamples = rec_folds,
    grid = lambda_grid
  )

# what is the best value of lambda?
top_acc  <- show_best(rec_res, metric = "accuracy")
best_acc <- select_best(rec_res, metric = "accuracy")
final_logit_lasso <- finalize_workflow(rec_wf,
                                       best_acc
)
best_acc

print('*********** LOGISTIC REGRESSION **************')
logit_test <- last_fit(final_logit_lasso,income_split) %>%
  collect_metrics()

logit_test %>% print(n = 1)
top_acc %>% print(n = 1)

# combine results into a nice tibble (for later use)
logit_ans <- top_acc %>% slice(1)
logit_ans %<>% left_join(logit_test %>% slice(1),by=c(".metric",".estimator")) %>%
  mutate(alg = "logit") %>% select(-starts_with(".config"))



#####################
# tree model
#####################
print('Starting TREE')
# set up the task and the engine
tune_tree_spec <- decision_tree(
  min_n = tune(), # tuning parameter
  tree_depth = tune(), # tuning parameter
  cost_complexity = tune(), # tuning parameter
) %>% 
  set_engine("rpart") %>%
  set_mode("classification")

# define a set over which to try different values of tuning parameters
tree_grid <- grid_latin_hypercube(
  min_n(),
  tree_depth(),
  cost_complexity(),
  size = 20
)

# 3-fold cross-validation
tree_folds <- vfold_cv(income_train, v = 3)

# Workflow
tree_wf <- workflow() %>%
  add_model(tune_tree_spec) %>%
  add_formula(high.earner ~ education + marital.status + race + workclass + occupation + relationship + sex + age + capital.gain + capital.loss + hours)

# Tuning results
tree_res <- tree_wf %>%
  tune_grid(
    resamples = tree_folds,
    grid = tree_grid
  )

# what are the best values of the tuning parameters?
top_acc_tree  <- show_best(tree_res, metric = "accuracy")
best_acc_tree <- select_best(tree_res, metric = "accuracy")
final_tree <- finalize_workflow(tree_wf,
                                best_acc_tree
)

print('*********** DECISION TREE **************')
tree_test <- last_fit(final_tree,income_split) %>%
  collect_metrics()

tree_test %>% print(n = 1)
top_acc_tree %>% print(n = 1)

# combine results into a nice tibble (for later use)
tree_ans <- top_acc_tree %>% slice(1)
tree_ans %<>% left_join(tree_test %>% slice(1),by=c(".metric",".estimator")) %>%
  mutate(alg = "tree") %>% select(-starts_with(".config"))

#####################
# k-nearest neighbors
#####################
print('Starting KNN')
# set up the task and the engine
tune_knn_spec <- nearest_neighbor(weight_func = tune(), # tuning parameter
                                  neighbors = tune() # tuning parameter
) %>% 
  set_engine("kknn") %>%
  set_mode("classification")

# define a set over which to try different values of tuning parameters
knn_grid <- grid_latin_hypercube(
  weight_func(),
  neighbors(),
  size = 20
)

# 3-fold cross-validation
knn_folds <- vfold_cv(income_train, v = 3)

# Workflow
knn_wf <- workflow() %>%
  add_model(tune_knn_spec) %>%
  add_formula(high.earner ~ education + marital.status + race + workclass + occupation + relationship + sex + age + capital.gain + capital.loss + hours)

# Tuning results
knn_res <- knn_wf %>%
  tune_grid(
    resamples = knn_folds,
    grid = knn_grid
  )

# what are the best values of the tuning parameters?
top_acc_knn  <- show_best(knn_res, metric = "accuracy")
best_acc_knn <- select_best(knn_res, metric = "accuracy")
final_knn <- finalize_workflow(knn_wf,
                               best_acc_knn
)

print('*********** K-NEAREST NEIGHBORS **************')
knn_test <- last_fit(final_knn,income_split) %>%
  collect_metrics()

knn_test %>% print(n = 1)
top_acc_knn %>% print(n = 1)

# combine results into a nice tibble (for later use)
knn_ans <- top_acc_knn


#####################
# SVM
#####################
print('Starting SVM')
# set up the task and the engine
tune_svm_spec <- svm_poly(
  degree = tune(), # tuning parameter: degree of polynomial kernel
  scale = tune(), # tuning parameter: scaling factor for kernel
  cost = tune(), # tuning parameter: cost of violation
  epsilon = 0.1 # epsilon for epsilon-insensitive loss function
) %>% 
  set_engine("LiblineaR") %>%
  set_mode("classification")

# define a grid over which to try different values of tuning parameters
svm_grid <- grid_tune(
  degree = seq(1, 5, by = 1), # try degrees from 1 to 5
  scale = seq(0.1, 1, by = 0.1), # try scaling factors from 0.1 to 1
  cost = 10^seq(-3, 3, by = 1) # try costs from 0.001 to 1000
)

# Workflow
svm_wf <- workflow() %>%
  add_model(tune_svm_spec) %>%
  add_formula(high.earner ~ education + marital.status + race + workclass + occupation + relationship + sex + age + capital.gain + capital.loss + hours)

# Tuning results
svm_res <- svm_wf %>%
  tune_grid(
    resamples = rec_folds,
    grid = svm_grid
  )

# what is the best combination of tuning parameters?
top_acc_svm <- show_best(svm_res, metric = "accuracy")
best_acc_svm <- select_best(svm_res, metric = "accuracy")
final_svm <- finalize_workflow(svm_wf,
                               best_acc_svm
)
print('*********** SUPPORT VECTOR MACHINE (SVM) **************')
svm_test <- last_fit(final_svm, income_split) %>%
  collect_metrics()

svm_test %>% print(n = 1)
top_acc_svm %>% print(n = 1)

# combine results into a nice tibble (for later use)
svm_ans <- top_acc_svm %>% slice(1)
svm_ans %<>% left_join(svm_test %>% slice(1),by=c(".metric",".estimator")) %>%
  mutate(alg = "svm") %>% select(-starts_with(".config"))



#####################
# combine answers
#####################
all_ans <- bind_rows(logit_ans,tree_ans,nnet_ans,knn_ans,svm_ans)
datasummary_df(all_ans %>% select(-.metric,-.estimator,-mean,-n,-std_err),output="markdown") %>% print
