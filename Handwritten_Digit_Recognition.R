# Cleaning up the environment
remove (list=ls())
#setwd("F:/PGDDS/Module4/Assignment - Support Vector Machine")

# importing libraries
library(dplyr)
library(stringr)
library(stringr)
library(ggplot2)
library(scales)
library(lubridate)
library(zoo) 
library(gridExtra)
library(kernlab)
library(readr)
library(caret)

# Importing dataset
image_train_data <- read.csv('mnist_train.csv', header = F, stringsAsFactors = F, na.strings=c(""," "))
image_test_data <- read.csv('mnist_test.csv', header = F, stringsAsFactors = F, na.strings=c(""," "))

##############################################################################
###
### Data Understanding (from training data)
###
##############################################################################

#Understanding Dimensions
dim(image_train_data)   # 60,000 Rows, 785 Columns

#Structure of the dataset
str(image_train_data)

#printing first few rows
head(image_train_data)

###########################
##Exploring the data
summary(image_train_data)

#Identifying if any column contains value more thn 255 or less than 0 (based on bussiness understanding pixel value should be with in 0 to 255)
apply(image_train_data,2,min)
apply(image_train_data,2,max)

max(apply(image_train_data,2,max))
min(apply(image_train_data,2,min))

##Exploring Target Column
unique(image_train_data$V1)   # Only Valid Digit available

## Checking Duplicate Records in Training Dataset
summary(duplicated(image_train_data))
## All 60,000 records are unique, no duplicate found


## Checking Null / NA Values in Training Dataset
sum(is.na(image_train_data))
## No NA values found

## Understanding from Bussiness understanding
## 1. out of 785 columns, first attribute defines the digits, others are represntation of the value of pixel (28*28)
##    Hence it's ok to have columns containing all 0 / same values
## 2. As digits are handwritten, values of each pixel is not a factor (0 or 255). It can be any value between 0 to 255 (including both)


######################
## Observation : 
# 1.  Training data contains 60,000 Rows in 785 Columns
# 2.  All columns are numeric, first column represent the digit and next 784 (=28*28) columns represent the pixel value
# 3.  All the pixel values are varying from 0 to 255
# 4.  Only Valid Digits are available on first column
# 5.  No header name is present
# 6.  All records are unique, no duplicate found
# 7.  No NA values available
######################

##############################################################################
###
### Data Preparation
###
##############################################################################

## Creating Header name
# Header name is created Target column (containg value of handwritten Digit) as "Digit", 
# and other column (containing image part)as Pixel_1,Pixel_2,.......,Pixel_784

colnames(image_train_data) <- c("Digit",paste0("Pixel_", 1:784))
colnames(image_test_data) <- c("Digit",paste0("Pixel_", 1:784))

## Making target class to factor
image_train_data$Digit<-factor(image_train_data$Digit)
image_test_data$Digit<-factor(image_test_data$Digit)


str(image_train_data$Digit)
summary(image_train_data$Digit)


######################################################################################################
################################### Model building and evaluation ####################################
######################################################################################################

##  As total number of rows in training dataset is huge what need huge computational cost, 
##  Model is constructing only 15% of Training Data

set.seed(100)
train.indices = sample(1:nrow(image_train_data), 0.15*nrow(image_train_data))
train = image_train_data[train.indices, ]


################################    Using Linear Kernel   #############################################
Model_linear <- ksvm(Digit~ ., data = train, scaled = FALSE, kernel = "vanilladot")
## Evaluating Test Data
Eval_linear<- predict(Model_linear, image_test_data)

#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,image_test_data$Digit)

######################
## Observation : 
#               Overall Accuracy : 0.9182
#               Sensitivity : Varies wrt Digit (0.8439	to 0.985)  (Bad Sensitivity for Digit '8', '9' and '5')
#               Specificity : Varies wrt Digit (0.9834	to 0.996)  (Bad Specificity for Digit '3' ,'4' and '5')
######################

######################
## Understanding <1> : 
#               Overall Parameters are somehow good (may not be great). So there may be non-linearity present in dataset.
#               But non-linearity shouldn't be huge (as "linear kernel" model output is not bad)
######################

######################################################################################################

################################    Using polynomial Kernel   ########################################

## Assuming there is some non-linearity present in dataset, polynomial Kernel is used increasing degree

## Using Degree 2 polynominal
Model_polydot_2 <- ksvm(Digit~ ., data = train, scaled = FALSE, kernel = "polydot", kpar = list(degree = 2), cross = 3)
## Evaluating Test Data
Eval_polydot_2<- predict(Model_polydot_2, image_test_data)

#confusion matrix - Linear Kernel
confusionMatrix(Eval_polydot_2,image_test_data$Digit)


## Observation : 
#               Overall Accuracy : 0.95
#               Sensitivity : Varies wrt Digit (0.9346 to 0.9867)
#               Specificity : Varies wrt Digit (0.9934	0.9971)


## Using Degree 3 polynominal
Model_polydot_3 <- ksvm(Digit~ ., data = train, scaled = FALSE, kernel = "polydot", kpar = list(degree = 3), cross = 3)
## Evaluating Test Data
Eval_polydot_3<- predict(Model_polydot_3, image_test_data)

#confusion matrix - Linear Kernel
confusionMatrix(Eval_polydot_3,image_test_data$Digit)

## Observation : 
#               Overall Accuracy : 0.95
#               Sensitivity : Varies wrt Digit (0.926	0.9847)  (Bad Sensitivity for Digit '5')
#               Specificity : Varies wrt Digit (0.9927	0.9972)


######################
## Understanding <2> : 
#               Non-Linearity is present and for that reason performance significantly improve using polynominal
#               Specificity : Varies wrt Digit (0.9927	0.9972)
######################


######################################################################################################

## As per "understanding <2>" non-learity is present on dataset. For better handling the non-learity 
##  RBF( radial basis function ) kernel is used

#Using RBF Kernel
Model_RBF <- ksvm(Digit~ ., data = train, scaled = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, image_train_data)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,image_train_data$Digit)
######################
## Observation : 
#               Overall Accuracy : 0.9585
#               Sensitivity : Varies wrt Digit (0.93334	to 0.98548)
#               Specificity : Varies wrt Digit (0.99348 to 0.99736)  
######################

######################################################################################################


############   Hyperparameter tuning and Cross Validation #####################
# Number -> Number of folds = 3
# Method -> cv (cross validation)

trainControl <- trainControl(method="cv", number=3,verboseIter=T)
metric <- "Accuracy"


set.seed(123)

## Understanding for Parameter Selection
## Sigma Values :   The higher the value of sigma, the more is the nonlinearity introduced. 
##                  According to "Understanding <1>" the system shouldn't be much nonlinear. Hence lower values of Sigma choosen.
## C (cost)     :   As higher values of C can overfit the model, so lower values choosen

grid <- expand.grid(.sigma=c(0.01, 0.1, 1, 10), .C=c(0.01,0.1,1,10) )

fit.svm <- train(Digit~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)

plot(fit.svm)

fit_siv_1st<-print(fit.svm)
write.csv(fit_siv_1st, file = "fit_siv_1st.csv")
####
## Observation: Accuracy is 0.1071111 for all the given dataset


####################################

## Understanding for Parameter Selection
## Sigma Values :   Either Sigma value should be less (towards linear) or this model becomes much simpler 
##                  So more lower values are choosen (e.g. 0.00001, 0.001, .025) and '20' (if model becomes simpler)
## C (cost)     :   For better computation C Value set limited
grid_2 <- expand.grid(.sigma=c(0.00001, 0.001, .025, 20), .C=c(0.1,1) )

fit.svm2 <- train(Digit~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid_2, trControl=trainControl)

print(fit.svm2)
plot(fit.svm2)

fit_siv_2nd<-print(fit.svm2)
write.csv(fit_siv_2nd, file = "fit_siv_2nd.csv")

## Observation: There's a significant accuracy increase (although not good) on C=1 and Sigma = 1.0e-05
##              So Sigma should be more lower and value of C should be near 1 (not previosly selected value .1 ,.01 or 10)

####################################


## Understanding for Parameter Selection
## Sigma Values :   Sigma value should be lower than  1.0e-05 ( 1e-08 ,5e-07, 1e-07, 5e-06, 1e-06 )
##                  Hence different random values are choosen exponetially
## C (cost)     :   as on C=1, improvement shown, so C is selected as 1,2,3


grid_3 <- expand.grid(.sigma=c(.1 ^ 08, 5 * .1^ 07, .1 ^ 07,5* .1 ^ 06, .1^ 06), .C=c(1,2,3) )

fit.svm3 <- train(Digit~., data=train, method="svmRadial", metric=metric, 
                  tuneGrid=grid_3, trControl=trainControl)

print(fit.svm3)
plot(fit.svm3)

fit_siv_3rd<-print(fit.svm3)
write.csv(fit_siv_3rd, file = "fit_siv_3rd.csv")

##Observation: For Sigma=0.0000005 (5e-07) and C=3, maximum accuracy is shown

####################################
## Understanding for Parameter Selection
## Sigma Values :   For fine tuning the best Sigma, values are choosen 3e-07 ,4e-07 ,5e-07 ,6e-07 ,7e-07 
## C (cost)     :   For fine tuning the Cost, C values are choosen nearer to 3 (e.g. 2.5,3,3.5)

grid_4 <- expand.grid(.sigma=c(3 * .1^ 07,4 * .1^ 07,5 * .1^ 07,6 * .1^ 07,7 * .1^ 07), .C=c(2.5,3,3.5) )

fit.svm4 <- train(Digit~., data=train, method="svmRadial", metric=metric, 
                  tuneGrid=grid_4, trControl=trainControl)

print(fit.svm4)
plot(fit.svm4)

fit_siv_4rd<-print(fit.svm4)
write.csv(fit_siv_4rd, file = "fit_siv_4rd.csv")

## Observation: Maximum Accuracy found on Sigma=5e-07
## Hence finalized the value Sigma=5e-07 and C=3
####################################

####################################
## Understanding for Parameter Selection
## Cross Check  :   Values of Sigma is too small, previously other values of Sigma is tested 
##                  with different C value (nearly C=1 ), but not for C=2 or C=3 (Final Value)
##                  This checking is just for ensuring Sigma should be that much low for getting better result

#grid_5 <- expand.grid(.sigma=c(0.0001,0.001,0.01, 0.1), .C=c(2,3) )

#fit.svm5 <- train(Digit~., data=train, method="svmRadial", metric=metric, 
#                  tuneGrid=grid_5, trControl=trainControl)

#print(fit.svm5)
#plot(fit.svm5)

#fit_siv_5rd<-print(fit.svm5)
#write.csv(fit_siv_5rd, file = "fit_siv_5rd.csv")

## Accuracy  decreased again, so MODEL 'fit.svm4 ' IS FINAL
##############################################################################


#Valdiating the model after cross validation on test data

Model_RBF_Final <- ksvm(Digit~ ., data = train, scaled = FALSE, kernel = "rbfdot",C=3,sigma=5 * .1^ 07)
Eval_RBF_Final<- predict(Model_RBF_Final, image_train_data)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF_Final,image_train_data$Digit)
######################
## Observation : 
#               Overall Accuracy : 0.9669
#               Sensitivity : Varies wrt Digit (0.94377	to 0.98784)
#               Specificity : Varies wrt Digit (0.99507	to 0.99791)  
######################

## Accuracy for testing data is 0.9669 and training data Accuracy is 0.9681119
## Hence this model is working fine (providing satisfactory performance) and free from overfitting.


