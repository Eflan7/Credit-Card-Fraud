install.packages("tidyverse")
library(tidyverse)

install.packages("h2o")
library("h2o")

install.packages("rio")
library(rio)

install.packages("doParallel")
library(doParallel)

install.packages("viridis")
library(viridis) 

install.packages("RColorBrewer")
library(RColorBrewer) 

install.packages("ggthemes")
library(ggthemes)

install.packages("knitr")
library(knitr) 

install.packages("caretEnsemble")
library(caretEnsemble) 

install.packages("caret")
library(caret) 

install.packages("plotly")
library(plotly)  

install.packages("lime")
library(lime) 

install.packages("dplyr")
library(dplyr)

install.packages("plotROC")
library(plotROC)

install.packages("pROC")
library(pROC) 

install.packages("h2o")
library("h2o")


localH2O = h2o.init(ip = 'localhost', port = 54321, nthreads = -1,max_mem_size = "8G")

no_cores <- detectCores() - 1 
cl<-makeCluster(no_cores)
registerDoParallel(cl)


#Import dataset
creditcard <- read.csv("creditcardtransactions.csv", header = TRUE)

#old datafram
cc_fraud_h2o <- as.h2o(creditcard)

#New data frame

dataNew <- (creditcard)

#EDA will go in here
print(str(creditcard))

print(table(creditcard$Amount))
print(table(creditcard$Time))
print(table(creditcard$Class))
#do a barplot for the above class (ggplot)






#relationship between 


#dont use not enough insight.
ggplot(creditcard,aes(x="",y=V1,fill=Class))+geom_boxplot()+labs(x="V1",y="") 

#distribution of transaction amounts with respect to class# (do all variables) # dont use this in report
ggplot(creditcard,aes(x = Amount)) + geom_histogram(color = "red", fill = "black", bins = 10)

#use this in report
ggplot(creditcard, aes(x =Time,fill = Class))+ geom_histogram(bins = 30)+  facet_wrap( ~ Class, scales = "free", ncol = 2)

#Plot the distrubtion?


#Pairs to remove time from, graph is not suitable for report
pairs(creditcard[, -c("Time")])
pairs(creditcard[, !(names(creditcard)) %in% c("Time")])





# data quality checks, remove highly correlated columns
# select all numeric columns
dataNum <- creditcard %>% select_if(is.numeric)

# remove rows with NA values (NA doesn't work well with correlation matrix)
dataNum <- dataNum[complete.cases(dataNum), ]

# remove columns with standard deviation = 0
dataNum <- dataNum[, apply(dataNum, 2, sd, na.rm=TRUE) != 0]

# get correlation matrix
corrData <- cor(dataNum[, !names(dataNum) %in% c("TARGET")])

# find highly correlated values, exact = TRUE recalculates correlation after
# removing one variable. Use cutoff of 0.7
correlated <- findCorrelation(corrData, cutoff = 0.7, exact = TRUE)
correlated <- sort(correlated)

# remove highly correlated numeric columns , no highly correlated columns found
reducedData <- dataNum
head(reducedData)

# combine reducedData with other character columns
dataChar <- dataNew %>% select_if(is.character)
dataChar <- dataChar[row.names(reducedData), ]
cleanData <- cbind(reducedData, dataChar)


cc_fraud_h2o <- as.h2o(cleanData)

splits <- h2o.splitFrame(cc_fraud_h2o,ratios = c(0.6, 0.2), seed = 148) 

train <- splits[[1]] 
validation <- splits[[2]] 
test <- splits[[3]] 
target <- "Class"
features <- setdiff(colnames(train), target)

# this will be used  can be dealt with
train$Class <- as.factor(train$Class) 



#code from 18/03/20
#-------------------------------------------------------------------------------
# function to ...
# explain the inputs and what they mean ...
# explain the output/return value
# rerun the function body every time you make changes to it.
#-------------------------------------------------------------------------------



model_function <- function(layers = NULL, activate = "") {
    
model_one = h2o.deeplearning (x = features, y = "Class", training_frame = train,
                                  balance_classes=T,
                                  reproducible = TRUE,
                                  seed = 148,
                                  hidden = layers,
                                  epochs = 50, 
                                  activation = activate, 
                                  validation_frame = validation)
    
train_features <- h2o.deepfeatures(model_one, train, layer = 3) %>%  
as.data.frame() %>%  mutate(Class = as.vector(train$Class)) %>% as.h2o()

features2 <- setdiff(colnames(train_features), "Class")
train_features$Class <- as.factor(train_features$Class)
    
    # with balance class
    model_two <- h2o.deeplearning(y = "Class", x = features2, 
                                  training_frame = train_features, balance_classes = TRUE,
                                  hidden = layers, activation = activate,
                                  reproducible = TRUE, epochs = 50,
                                  validation_frame = validation)
    return(model_two)
  }

tanh <- model_function(layers = c(10,10,10), activate = "Tanh")
h2o.performance(tanh) # focus on confusion matrix
plot(h2o.performance(tanh))

#Dont use this code, test is unbalanced
#dl_perf1 <- h2o.performance(model = tanh, newdata = test)
#dl_perf1



tanhh2 <- model_function(layers = c(5,5,5), activate = "Tanh")
h2o.performance(tanhh2)
plot(h2o.performance(tanh2))


tanhh3 <- model_function(layers = c(8,8,8), activate = "Tanh")
h2o.performance(tanhh3)
plot(h2o.performance(tanhh3))

Maxout <- model_function(layers = c(10,10,10), activate = "Maxout")
h2o.performance(Maxout)
plot(h2o.performance(Maxout))

Maxout2 <- model_function(layers = c(5,5,5), activate = "Maxout")
h2o.performance(Maxout2)
plot(h2o.performance(Maxout2))

Maxout3 <- model_function(layers = c(8,8,8), activate = "Maxout")
h2o.performance(Maxout3)
plot(h2o.performance(Maxout3))

Maxoutwith <- model_function(layers = c(10,10,10), activate = "MaxoutwithDropout")
h2o.performance(Maxoutwith)
plot(h2o.performance(Maxoutwith))

Maxoutwith2 <- model_function(layers = c(5,5,5), activate = "MaxoutwithDropout")
h2o.performance(Maxoutwith2)
plot(h2o.performance(Maxoutwith2))

Maxoutwith3 <- model_function(layers = c(8,8,8), activate = "MaxoutwithDropout")
h2o.performance(Maxoutwith3)
plot(h2o.performance(Maxoutwith3))

save.image("~/Draft of code run creditcard.RData")




#remove the below and save to draft file














#code from code 26/02, models


creditcard <- read.csv("creditcardtransactions.csv", header = TRUE)
cc_fraud_h2o <- as.h2o(creditcard)

splits <- h2o.splitFrame(cc_fraud_h2o,ratios = c(0.6, 0.2), seed = 148) 

train <- splits[[1]] 
validation <- splits[[2]] 
test <- splits[[3]] 
target <- "Class"
features <- setdiff(colnames(train), target)

# this will be used for classification and class imbalance can be dealt with
train$Class <- as.factor(train$Class) 

model_one = h2o.deeplearning (x = features, y = "Class", training_frame = train,
                              balance_classes=T,
                              reproducible = TRUE,
                              seed = 148,
                              hidden = c(10,10,10),
                              epochs = 1,
                              activation = "Tanh", 
                              validation_frame = validation)

summary(model_one)
h2o.performance(model_one) # focus on confusion matrix
plot(h2o.performance(model_one)) # plot AUC curve

#-------------------------------------------------------------------------------
# get features from first model to be used in second model

train_features <- h2o.deepfeatures(model_one, train, layer = 2) %>%  
  as.data.frame() %>%  mutate(Class = as.vector(train$Class)) %>% as.h2o()

target2 = "Class"
features2 <- setdiff(colnames(train_features), target2)

# without balance class
train_features$Class <- as.factor(train_features$Class)
model_Tahn_two <- h2o.deeplearning(y = target2, x = features2, 
                                   training_frame = train_features, 
                                   nfolds = 0, reproducible = TRUE, epochs = 1)

h2o.confusionMatrix(model_Tahn_two)

# with balance class
model_Tahn_two_balance <- h2o.deeplearning(y = target2, x = features2, 
                                           training_frame = train_features, balance_classes = TRUE,
                                           nfolds = 0, reproducible = TRUE, epochs = 1)

h2o.confusionMatrix(model_Tahn_two_balance) # compare this with earlier matrix

h2o.mse(model_Tahn_two)

dl_perf1 <- h2o.performance(model = model_Tahn_two, newdata = test)
h2o.mse(dl_perf1)

help(h2o.performance)


#testing of model with Maxout
model_one_Max = h2o.deeplearning (x = features, y = "Class", training_frame = train,
                              balance_classes=T,
                              reproducible = TRUE,
                              seed = 148,
                              hidden = c(10,10,10),
                              epochs = 1,
                              activation = "Maxout", 
                              validation_frame = validation)

summary(model_one_Max)
h2o.performance(model_one_Max) # focus on confusion matrix
plot(h2o.performance(model_one_Max)) # plot AUC curve


#testing of model with Maxoutwithdropout
model_one_Maxdrop = h2o.deeplearning (x = features, y = "Class", training_frame = train,
                                  balance_classes=T,
                                  reproducible = TRUE,
                                  seed = 148,
                                  hidden = c(10,10,10),
                                  epochs = 1,
                                  activation = "MaxoutwithDropout", 
                                  validation_frame = validation)

summary(model_one_Maxdrop)
h2o.performance(model_one_Maxdrop) # focus on confusion matrix
plot(h2o.performance(model_one_Maxdrop)) # plot AUC curve


#




































































# Moving dataframe into H20 

creditcard_h2o <- as.h2o(creditcard)    

split_creditcard <- h2o.splitFrame(creditcard_h2o,ratios = c(0.6, 0.2), seed = 148) 

train <- split_creditcard[[1]] 
valid <- split_creditcard[[2]] 
testing <- split_creditcard[[3]]


splits <- h2o.splitFrame(cc_fraud_h2o,ratios = c(0.6, 0.2), seed = 148) 

train <- splits[[1]] 
validation <- splits[[2]] 
test <- splits[[3]] 
target <- "Class"
features <- setdiff(colnames(train), target)

str(train)


aim <- "Class" 
aim <- setdiff(colnames(train), aim )


#epoch run at one due to wait
model_Tahn = h2o.deeplearning(x = features, training_frame = train, 
                             reproducible = TRUE, seed= 123,  
                             hidden = c(10,10,10), epochs = 50, activation = "Tanh", 
                             validation_frame= valid, autoencoder = TRUE)
summary(model_Tahn)   
h2o.predict(model_Tahn, testing)


training_hf <- h2o.deepfeatures(model_Tahn, train, layer = 2)
print(training_hf)

training_f <- h2o.deepfeatures(model_Tahn, valid, layer = 3)
print(training_f)


Slice_ae <- setdiff(colnames(training_f), aim)
print(Slice_ae)

model_Tahn_two <- h2o.deeplearning(y= aim, x= Slice_ae, training_frame = training_f, model_id=Null, nfolds=5,
                                 reproducible=TRUE, balance_classes=TRUE)


#second model
model_two <- h2o.deeplearning(y = aim, 
                              x = Slice_ae,
                              training_frame = training_f,
                              reproducible = TRUE,
                              balance_classes = TRUE,
                              ignore_const_cols = FALSE,
                              seed = 123,
                              hidden = c(10, 5, 10),
                              epochs = 50,
                              activation = "Tanh") 
                              
                              
#second attempted


cc_fraud_h2o <- as.h2o(creditcardtran)

splits <- h2o.splitFrame(cc_fraud_h2o,ratios = c(0.6, 0.2), seed = 148) 

train <- splits[[1]] 
validation <- splits[[2]] 
test <- splits[[3]] 
target <- "Class"
features <- setdiff(colnames(train), target)
              
model_one = h2o.deeplearning (x = features, training_frame = train,
autoencoder = TRUE,
reproducible = TRUE,
seed = 148,
hidden = c(10,10,10),
epochs = 1,
activation = "Tanh", 
validation_frame = test) 

h2o.saveModel(model_one, path="model_one", force = TRUE) 
model_one<h2o.loadModel("/home/sunil/model_one/DeepLearning_model_R_1544970545051_1") 
print(model_one) 


test_autoencoder <- h2o.predict(model_one, test) 

train_features <- h2o.deepfeatures(model_one, train, layer = 2) %>%  
as.data.frame() %>%  mutate(Class = as.vector(train[, 31]))

print(train_features%>%head(3))

# printing the reduced data represented in layer2 print(train_features%>%head(3)) 

help("h2o.deeplearning")
help("setdiff")










model_Relu = h2o.deeplearning(x = features, training_frame = training, 
                              reproducible = TRUE, seed= 123,  
                              hidden = c(10,10,10), epochs = 50, activation = "Relu", 
                              validation_frame= valid, autoencoder = TRUE)
summary(model_Relu) 
h2o.predict(model_Relu, testing)



model_Maxout = h2o.deeplearning(x = features, training_frame = training, 
                              reproducible = TRUE, seed= 123,  
                              hidden = c(10,10,10), epochs = 50, activation = "Maxout", 
                              validation_frame= valid, autoencoder = TRUE)

summary(model_Maxout)

help(maxout)

#code from NIk
model_Maxout = h2o.deeplearning(x = features, y = target, training_frame = train, 
                                reproducible = TRUE, seed= 123,  
                                hidden = c(10,10,10), epochs = 1, activation = "Maxout", 
                                validation_frame= test, autoencoder = FALSE)







































errors <- h2o.anomaly(model_one, train , per_feature = FALSE)

#Does this need to be save? Yes. so it does not need to be retrained
h20.saveModel(model_full)

model_path <- h2o.saveModel(model_full, path=model_full, force=T)

h2o.saveModel



#predict function
testing_ae <- h2o.predict(model_full, test)

help(save.model)






# based on plot above

as.data.frame(train)
errors <- as.data.frame(errors)

# based on plot above

row_outliers <- which (errors > 0.0) 

train[row_outliers,]

help("which")

hist(errors, breaks = 50 [,1])

ggplot(errors(, x=)) + geom_histogram (breaks = 30)

ggplot(errors,aes(x = errors[,1])) + geom_histogram(color = "red", fill = "black", bins = 50)

ggplot(errors, aes(x=errors[,1])) + geom_histogram(bins = 50)

# do this graph in Orange
write.csv(errors,"errors.csv", row.names = FALSE)

#Run full model

model_full = h2o.deeplearning(x = features, training_frame = creditcard_h2o, 
                              reproducible = TRUE, seed= 148,  
                              hidden = c(10,10,10), epochs = 1, activation = "Tanh", 
                              
                              validation_frame= test, autoencoder = TRUE)
summary(model_full)

errors <- h2o.anomaly(model_full, creditcard_h2o , per_feature = FALSE)
errors <- as.data.frame(errors)
row_outliers <- which (errors > 0.01) 
creditcard_h2o [row_outliers,]

plot(errors)


model_full = h2o.deeplearning(x = features, training_frame = creditcard_h2o, 
                              reproducible = TRUE, seed= 148,  
                              hidden = c(10,10,10), epochs = 1, activation = "Tanh", 
                              
                              validation_frame= test, autoencoder = TRUE)
summary(model_full)


help(h2o.deeplearning) 




#code not to be used #code not to be used.


#second attempted


cc_fraud_h2o <- as.h2o(creditcardtran)

splits <- h2o.splitFrame(cc_fraud_h2o,ratios = c(0.6, 0.2), seed = 148) 

train <- splits[[1]] 
validation <- splits[[2]] 
test <- splits[[3]] 
target <- "Class"
features <- setdiff(colnames(train), target)

model_one = h2o.deeplearning (x = features, training_frame = train,
                              autoencoder = TRUE,
                              reproducible = TRUE,
                              seed = 148,
                              hidden = c(10,10,10),
                              epochs = 1,
                              activation = "Tanh", 
                              validation_frame = test) 

h2o.saveModel(model_one, path="model_one", force = TRUE) 
model_one<h2o.loadModel("/home/sunil/model_one/DeepLearning_model_R_1544970545051_1") 
print(model_one) 


test_autoencoder <- h2o.predict(model_one, test) 

train_features <- h2o.deepfeatures(model_one, train, layer = 2) %>%  
  as.data.frame() %>%  mutate(Class = as.vector(train[, 31]))

print(train_features%>%head(3))

# printing the reduced data represented in layer2 print(train_features%>%head(3)) 



#wont load data
#Dont use below
creditcard <- read.csv("creditcardtransactions.csv", header = TRUE)

library(corrplot)
raw.data <- read.csv("../input/creditcard.csv")
n <- colnames(raw.data)
cols <- n[!n %in% c("Class", "Time", "Amount")]
V <- raw.data[,cols]
correlations <- cor(V)
p <- corrplot(correlations, method="circle")


#Import dataset

creditcardtrans <-load(choose.files())


#wont load data
#Dont use below
creditcard <- read.csv("creditcardtransactions.csv", header = TRUE)
View(creditcard)



#New file


creditcard <- read.csv("creditcardtransactions.csv", header = TRUE)
cc_fraud_h2o <- as.h2o(creditcard)

splits <- h2o.splitFrame(cc_fraud_h2o,ratios = c(0.6, 0.2), seed = 148) 

train <- splits[[1]] 
validation <- splits[[2]] 
test <- splits[[3]] 
target <- "Class"
features <- setdiff(colnames(train), target)

# this will be used for classification and class imbalance can be dealt with
train$Class <- as.factor(train$Class) 

model_one = h2o.deeplearning (x = features, y = "Class", training_frame = train,
                              balance_classes=T,
                              reproducible = TRUE,
                              seed = 148,
                              hidden = c(10,10,10),
                              epochs = 1,
                              activation = "Tanh", 
                              validation_frame = validation)

summary(model_one)
h2o.performance(model_one) # focus on confusion matrix
plot(h2o.performance(model_one)) # plot AUC curve

#-------------------------------------------------------------------------------
# get features from first model to be used in second model

train_features <- h2o.deepfeatures(model_one, train, layer = 2) %>%  
  as.data.frame() %>%  mutate(Class = as.vector(train$Class)) %>% as.h2o()

target2 = "Class"
features2 <- setdiff(colnames(train_features), target2)

# without balance class
train_features$Class <- as.factor(train_features$Class)
model_Tahn_two <- h2o.deeplearning(y = target2, x = features2, 
                                   training_frame = train_features, 
                                   nfolds = 0, reproducible = TRUE, epochs = 1)

h2o.confusionMatrix(model_Tahn_two)

# with balance class
model_Tahn_two_balance <- h2o.deeplearning(y = target2, x = features2, 
                                           training_frame = train_features, balance_classes = TRUE,
                                           nfolds = 0, reproducible = TRUE, epochs = 1)

h2o.confusionMatrix(model_Tahn_two_balance) # compare this with earlier matrix

h2o.mse(model_Tahn_two)

dl_perf1 <- h2o.performance(model = model_Tahn_two, newdata = test)
h2o.mse(dl_perf1)