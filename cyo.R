# 1. loading the required libraries

library(readxl)
library(caret)
library(class)
library(tidyverse)
library(rpart.plot)
library(randomForest)
library(rpart)
library(corrplot)

# 2. setting output format and reading in the dataset

options(dplyr.width = Inf)

setwd("C:\\data\\capstone\\cyo")

wine_data = read_csv("winequality-red.csv")
head(wine_data)

# 3. examining the range and distribution of wine quality

wine_data %>% group_by(quality) %>% summarize(count = n())

wine_data[12] %>% min()
wine_data[12] %>% max()

# 4. creating the normalization function that will be used to normalize the variables used for predicting in KNN

normalize = function(x){
  (x - mean(x))/sd(x)
}
# 5. creating the category variables which are to be predicted, based off of the quality variable

wine_data = wine_data %>% mutate(category = if_else(quality < 6, 0, ifelse(quality > 6, 2, 1)))

names(wine_data)

# 6. creating a histogram of all the qualities

hist(wine_data$quality, xlab = "quality", main = "Histogram of Wine Qualities")

# 7. creating a barplot of the three wine categories

df = wine_data
df = df %>% mutate(gb = if_else(category == 0, "bad", ifelse(category == 1, "medium", "good"))) 

gmb = df[14]
gmb = gmb %>% group_by(gb) %>% summarize(count = n())

M = c("bad", "medium", "good")

barplot(c(744, 638, 217), ylim = c(0, 800), names.arg = M, xlab = "category", ylab = "frequency", main = "Wine Category Chart")

gmb %>% mutate(percentage = 100*count/sum(count))

# 8. renaming some variable so they would have no spaces in the names

wine_data = wine_data %>% rename(fixed_acidity = `fixed acidity`, volatile_acidity = `volatile acidity`, citric_acid = `citric acid`, residual_sugar = `residual sugar`, free_sulfur_dioxide = `free sulfur dioxide`, total_sulfur_dioxide = `total sulfur dioxide`)

# 9. creating boxplots to see how wine category relates to the other variables

par(mfrow = c(3,2))

boxplot(fixed_acidity ~ category, data = wine_data, outline = FALSE)

boxplot(volatile_acidity ~ category, data = wine_data)

boxplot(citric_acid ~ category, data = wine_data)

boxplot(residual_sugar ~ category, data = wine_data)

boxplot(chlorides ~ category, data = wine_data)

boxplot(free_sulfur_dioxide ~ category, data = wine_data, outline = FALSE)

par(mfrow = c(3,2))

boxplot(total_sulfur_dioxide ~ category, data = wine_data)

boxplot(density ~ category, data = wine_data, outline = FALSE)

boxplot(pH ~ category, data = wine_data, outline = FALSE)

boxplot(sulphates ~ category, data = wine_data)

boxplot(alcohol ~ category, data = wine_data)

names(wine_data)

# 10. creating the correlation plot between variables

corrplot(cor(wine_data))

dim(wine_data)

# 11. looking at distribution of wine qualities

wine_data %>% group_by(quality) %>% summarize(count = n())

# 12. splitting up the normalized dataset into a training set and a testing set

set.seed(619, sample.kind="Rounding")
test_index <- createDataPartition(y = wine_data$quality, times = 1,
                                  p = 0.2, list = FALSE)

train_data = wine_data[-test_index,]
test_data = wine_data[test_index,]

# 13. normalizing all variables in the train set except quality and category

train_1_11 = train_data[,1:11]

train_1_11 = apply(train_1_11, MARGIN = 2, FUN = normalize)
train_12_13 = train_data[12:13]

train_data_norm = cbind(train_1_11, train_12_13)
head(train_data_norm)

# 14. normalizing all variables in the train set except quality and category

test_1_11 = test_data[,1:11]

test_1_11 = apply(test_1_11, MARGIN = 2, FUN = normalize)
test_12_13 = test_data[12:13]

test_data_norm = cbind(test_1_11, test_12_13)
head(test_data_norm)

# 15. checking things are as should

str(train_data_norm)
dim(train_data_norm)
names(train_data_norm)

str(test_data_norm)
dim(test_data_norm)
names(test_data_norm)

# 16. looking at the distribution table of the wine categories

wine_data %>% group_by(category) %>% summarize(count = n())

# 17. looking at correlations of normalized dataset and summary of the non-normalized dataset

cor(wine_data)
summary(wine_data)

# 18. checking dimensions of training set and testing set

dim(train_data)
dim(test_data)

# 19. checking class of category variable

class(test_data$category)

# 20. creating the range of nearest neighbors to be used in KNN

k = 1:20

# 21. calculating the accuracy for each amount of NN in the 1-20 range

accuracies = sapply(k, function(x){
  preds_knn = knn(train_data_norm, test_data_norm, train_data$category, k = x)
  confusionMatrix(preds_knn, factor(test_data$category))$overall['Accuracy']
})

# 22. looking at results from previous calculation and determining the amount of nearest neighbors that gives the highest accuracy 

df = data.frame(k, accuracies)
df %>% ggplot(aes(k, accuracies)) + geom_point()

df$accuracies %>% which.max()
df$accuracies %>% max()

# 23. predicting category values for the test set using KNN

preds_knn = knn(train_data_norm, test_data_norm, train_data$category, k = 1)

# 24. checking class of predicted values

class(preds_knn)

# 25. creating and plotting a decision tree model

fit_rpart <- rpart(category ~ fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol, method="class", data = train_data)

rpart.plot(fit_rpart)

# 26. creating predicted values for category in the test set using decision tree method

preds_rpart = predict(fit_rpart, test_data, type = "class")

# 27. checking to see that there are as many predictions as there are values to be predicted

length(preds_rpart)

# 28. converting the category variable to be a factor variable

str(train_data)
train_data = train_data %>% mutate(category = as.factor(category))
str(test_data)
test_data = test_data %>% mutate(category = as.factor(category))

# 29. creating the random forest model from the training set

fit_rf = randomForest(category ~ fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol, data = train_data, proximity = TRUE, method = "class")

# 30. predicting category values for the test set using random forest

preds_rf = predict(fit_rf, test_data, type = "class")

# 31. checking that there are 321 predicted values from using random forest

length(preds_rf)

# 32. checking to see RMSE from using random forest method

RMSE(as.numeric(preds_rf), as.numeric(test_data$category))

# 33. putting the actual values and the predicted values from random forest, KNN and decision tree into one table

table_preds = tibble(actual = test_data$category, preds_knn = preds_knn, preds_rpart = preds_rpart, preds_rf = preds_rf)

# 34. having the three methods vote to determine the most popular predicted value for each row in the test set 

most_voted = apply(table_preds, 1 , function(x) names(which.max(table(x))))
most_voted = as.factor(most_voted)

# 35. checking to see the popular vote results are of same length as the other ones

length(most_voted)

# 36. adding a column to the table with results from the ensemble method

table_preds = table_preds %>% mutate(preds_ensemble = most_voted)

# 37. changing all the actual and predicted values to numeric to later compute the RMSE  

table_preds = table_preds %>% mutate(actual = as.numeric(actual), preds_knn = as.numeric(preds_knn), preds_rpart = as.numeric(preds_rpart), preds_rf = as.numeric(preds_rf), preds_ensemble = as.numeric(preds_ensemble))

# 38. seeing results of the KNN predictions

table(table_preds$preds_knn, table_preds$actual)
confusionMatrix(as.factor(table_preds$preds_knn), factor(table_preds$actual))$overall['Accuracy']
RMSE(as.numeric(table_preds$preds_knn), as.numeric(table_preds$actual))

# 39. seeing results of the decision tree method predictions

table(table_preds$preds_rpart, table_preds$actual)
confusionMatrix(as.factor(table_preds$preds_rpart), factor(table_preds$actual))$overall['Accuracy']
RMSE(as.numeric(table_preds$preds_rpart), as.numeric(table_preds$actual))

# 40. seeing results of random forest predictions

table(table_preds$preds_rf, table_preds$actual)
confusionMatrix(as.factor(table_preds$preds_rf), factor(table_preds$actual))$overall['Accuracy']
RMSE(as.numeric(table_preds$preds_rf), as.numeric(table_preds$actual))

# 41. seeing results from the ensemble method

table(table_preds$preds_ensemble, table_preds$actual)
confusionMatrix(as.factor(table_preds$preds_ensemble), factor(table_preds$actual))$overall['Accuracy']
RMSE(as.numeric(table_preds$preds_ensemble), as.numeric(table_preds$actual))

# 42. putting the results of all the four algorithms together

results_table_knn = tibble(Method = "KNN", Accuracy = confusionMatrix(as.factor(table_preds$preds_knn), factor(table_preds$actual))$overall['Accuracy'], RMSE = RMSE(as.numeric(table_preds$preds_knn), as.numeric(table_preds$actual))) 

results_table_rpart = tibble(Method = "Decision Tree", Accuracy = confusionMatrix(as.factor(table_preds$preds_rpart), factor(table_preds$actual))$overall['Accuracy'], RMSE = RMSE(as.numeric(table_preds$preds_rpart), as.numeric(table_preds$actual)))

results_table_rf = tibble(Method = "Random Forest", Accuracy = confusionMatrix(as.factor(table_preds$preds_rf), factor(table_preds$actual))$overall['Accuracy'], RMSE = RMSE(as.numeric(table_preds$preds_rf), as.numeric(table_preds$actual)))

results_table_ensemble = tibble(Method = "Ensemble", Accuracy = confusionMatrix(as.factor(table_preds$preds_ensemble), factor(table_preds$actual))$overall['Accuracy'], RMSE = RMSE(as.numeric(table_preds$preds_ensemble), as.numeric(table_preds$actual)))

final_results_table = rbind(results_table_knn, results_table_rpart, results_table_rf, results_table_ensemble)
final_results_table

# 43. creating histograms for each of the initial variables

par(mfrow = c(3,2))

hist(wine_data$fixed_acidity, xlab = "Fixed Acidity", main = "Fixed Acidity Histogram")
hist(wine_data$volatile_acidity, xlab = "Volatile Acidity", main = "Volatile Acidity Histogram")
hist(wine_data$citric_acid, xlab = "Citric Acid", main = "Citric Acid Histogram")
hist(wine_data$residual_sugar, xlab = "Residual Sugar", main = "Residual Sugar Histogram")
hist(wine_data$chlorides, xlab = "Chlorides", main = "Chlorides Histogram")
hist(wine_data$free_sulfur_dioxide, xlab = "Free Sulfur Dioxide", main = "Free Sulfur Dioxide Histogram")

par(mfrow = c(3,2))

hist(wine_data$total_sulfur_dioxide, xlab = "Total Sulfur Dioxide", main = "Total Sulfur Dioxide Histogram")
hist(wine_data$density, xlab = "Density", main = "Density Histogram")
hist(wine_data$pH, xlab = "pH", main = "pH Histogram")
hist(wine_data$sulphates, xlab = "Sulphates", main = "Sulphates Histogram")
hist(wine_data$alcohol, xlab = "Alcohol", main = "Alcohol Histogram")
hist(wine_data$quality, xlab = "Quality", main = "Quality Histogram")


par(mfrow = c(3,2))





















#--------------------------------------------------------------------------------------------
# 44. rerunning similar KNN analysis but this time quality is to be the predicted variable rather than category

# 45. loading in data
wine_data = read_csv("winequality-red.csv")
head(wine_data)

# 46. examining the range and distribution of wine quality

wine_data %>% group_by(quality) %>% summarize(count = n())

wine_data[12] %>% min()
wine_data[12] %>% max()

# 47. creating the normalization function that will be used to normalize the variables used for predicting in KNN

normalize = function(x){
  (x - mean(x))/sd(x)
}

# 48. creating a histogram of all the qualities

hist(wine_data$quality, xlab = "quality", main = "Histogram of Wine Qualities")

# 49. renaming some variable so they would have no spaces in the names

wine_data = wine_data %>% rename(fixed_acidity = `fixed acidity`, volatile_acidity = `volatile acidity`, citric_acid = `citric acid`, residual_sugar = `residual sugar`, free_sulfur_dioxide = `free sulfur dioxide`, total_sulfur_dioxide = `total sulfur dioxide`)

# 50. splitting up the normalized dataset into a training set and a testing set

set.seed(619, sample.kind="Rounding")
test_index <- createDataPartition(y = wine_data$quality, times = 1,
                                  p = 0.2, list = FALSE)

train_data = wine_data[-test_index,]
test_data = wine_data[test_index,]

# 51. normalizing all variables in the train set except quality

train_1_11 = train_data[,1:11]

train_1_11 = apply(train_1_11, MARGIN = 2, FUN = normalize)
train_12 = train_data[12]

train_data_norm = cbind(train_1_11, train_12)
head(train_data_norm)

# 52. normalizing all variables in the test set except quality

test_1_11 = test_data[,1:11]

test_1_11 = apply(test_1_11, MARGIN = 2, FUN = normalize)
test_12 = test_data[12]

test_data_norm = cbind(test_1_11, test_12)
head(test_data_norm)

# 53. checking things are as should

str(train_data_norm)
dim(train_data_norm)
names(train_data_norm)

str(test_data_norm)
dim(test_data_norm)
names(test_data_norm)

# 54. checking dimensions of training set and testing set

dim(train_data)
dim(test_data)

# 55. creating the range of nearest neighbors to be used in KNN

k = 1:20

# 56. calculating the accuracy for each amount of NN in the 1-20 range

accuracies = sapply(k, function(x){
  preds_knn = knn(train_data_norm, test_data_norm, train_data$quality, k = x)
  confusionMatrix(preds_knn, factor(test_data$quality))$overall['Accuracy']
})

# 57. looking at results from previous calculation and determining the amount of nearest neighbors that gives the highest accuracy 

df = data.frame(k, accuracies)
df %>% ggplot(aes(k, accuracies)) + geom_point()

df$accuracies %>% which.max()
df$accuracies %>% max()








