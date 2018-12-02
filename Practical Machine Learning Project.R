## caret package installation
install.packages("caret")
install.packages("rpart.plot")
install.packages("RColorBrewer")
install.packages("rattle")
install.packages("randomForest")
install.packages('e1071', dependencies=TRUE)
install.packages("gbm")





## Loading package
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle) ## Rattle: A free graphical interface for data mining with R.
library(randomForest)



#loading and reading data 
pmlTrainData <- read.csv ("pml-training.csv", header = TRUE)
pmlTestData <- read.csv("pml-testing.csv", header = TRUE)


# Here we get the indexes of the columns having at least 90% of NA or blank values on the training dataset
trainColRemove <- which(colSums(is.na(pmlTrainData) |pmlTrainData=="")>0.9*dim(pmlTrainData)[1]) 
myTrainDataClean <- pmlTrainData[,-trainColRemove]
myTrainDataClean <- myTrainDataClean[,-c(1:7)]
dim(myTrainDataClean)


TestColToRemove <- which(colSums(is.na(pmlTestData) |pmlTestData=="")>0.9*dim(pmlTestData)[1]) 
myTestDataClean <- pmlTestData[,-TestColToRemove]
myTestDataClean <- myTestDataClean[,-1]
dim(myTestDataClean)

#partitioning
set.seed(12345)
inTrainData <- createDataPartition (y = myTrainDataClean$classe, p=0.7, list=FALSE)
myTrainData  <- myTrainDataClean[inTrainData, ];
myTestData  <- myTrainDataClean[-inTrainData, ];
dim(myTrainData); 
dim(myTestData);



# remove variables with nearly zero variance
nvrTrainData <- nearZeroVar(myTrainData, saveMetrics=TRUE)
myTrainData <- myTrainData[,nvrTrainData$nzv==FALSE]

nvrTestData <- nearZeroVar(myTestData,saveMetrics=TRUE)
myTestData <- myTestData[,nvrTestData$nzv ==FALSE]


myTrainData <- myTrainData[c(-1)]
dim(myTrainData)

cleanTrainData <- myTrainData
for(i in 1:length (myTrainData)) {
        if( sum( is.na (myTrainData [, i] ) ) / nrow (myTrainData) >= 0.7) {
            for(j in 1:length(cleanTrainData)) {
                  if( length( grep (names (myTrainData[i]), names(cleanTrainData)[j]) ) == 1)  {
                        cleanTrainData  <- cleanTrainData [ , -j]
                        }   
             } 
        }
}

myTrainData <- cleanTrainData
rm(cleanTrainData)


clean1 <- colnames(myTrainData)
clean2 <- colnames(myTrainData[, -58])  # remove the classe column
myTestData <- myTestData[clean1]         # allow only variables in myTesting that are also in mytraindata
myTestData2 <- myTestData [clean2]             # allow only variables in testing that are also in mytraindata

dim(myTestData2)



for (i in 1:length(myTestData2) ) {
  for(j in 1:length(myTrainData)) {
    if( length( grep(names(myTrainData[i]), names(myTestData2)[j]) ) == 1)  {
      class(myTestData2[j]) <- class(myTrainData[i])
    }      
  }      
}


plot(ClassTree$table, col = ClassTree$byClass, main = paste("Decision Tree Confusion Matrix: Accuracy =", 
                      round(ClassTree$overall['Accuracy'], 4)))
# To get the same class between testing and myTraining
myTestData2 <- rbind(myTrainData[2, -58] , myTestData2)
myTestData2 <- myTestData2[-1,]

#prediction with Decision tree
set.seed(12345)
ModelFit1 <- rpart(classe ~ ., data=myTrainData, method="class")
fancyRpartPlot(ModelFit1)


PredictModel1 <- predict(ModelFit1 , myTestData2, type = "class")
ClassTree <- confusionMatrix(PredictModel1, myTestData2$classe)
ClassTree

plot(ClassTree$table, col = ClassTree$byClass, 
     main = paste("Decision Tree Confusion Matrix: Accuracy =", round(ClassTree$overall['Accuracy'], 4)))

#Prediction with Random Forest
set.seed(12345)
ModelFit2 <- randomForest(classe ~ ., data=myTrainData)
PredictionModel2 <- predict(ModelFit2, myTestData2, type = "class")
RandomForestModel <- confusionMatrix(PredictionModel2, myTestData2$classe)
RandomForestModel 

plot(ModelFit2)

plot(RandomForestModel$table, col = ClassTree$byClass, 
     main = paste("Random Forest Confusion Matrix: Accuracy =", round(RandomForestModel$overall['Accuracy'], 4)))


#General Boost Regression
set.seed(12345)
FitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 1)

GRBFit <- train(classe ~ ., data=myTrainData, method = "gbm",
                 trControl = FitControl,
                 verbose = FALSE)


GBRModel <- GRBFit$finalModel

GBRPredictTest <- predict(GRBFit, newdata=myTestData2)
GBRAccuracy <- confusionMatrix(GBRPredictTest, myTestData2$classe)
GBRAccuracy 

plot(GRBFit , ylim=c(0.9, 1))