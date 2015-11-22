---
title: "Project Report - Practical Machine Learning"
author: "Vijay Anand"
date: "November 21, 2015"
output: html_document
---

##Introduction
The goal of this project is to predict the manner in which they did the exercise. 
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

###Getting and Cleaning Data

```r
##Load all neceassary Library.
set.seed(55555)
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(kernlab)
```

```
## Warning: package 'kernlab' was built under R version 3.2.2
```

```
## Warning in .recacheSubclasses(def@className, def, doSubclasses, env):
## undefined subclass "externalRefMethod" of class "kfunction"; definition not
## updated
```

```r
library(knitr)
library(rpart)
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.2.2
```

```r
library(RColorBrewer)
```

```
## Warning: package 'RColorBrewer' was built under R version 3.2.2
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.2.2
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(e1071)
```

```
## Warning: package 'e1071' was built under R version 3.2.2
```

```r
##Training and Testing data
trainURL<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
```

Download the testing and training data and read those files.

```r
Training <- read.csv(url(trainURL), na.strings=c("NA","#DIV/0!",""))
dim(Training)
```

```
## [1] 19622   160
```

```r
Testing <- read.csv(url(testURL), na.strings=c("NA","#DIV/0!",""))
dim(Testing)
```

```
## [1]  20 160
```

This training dataset can be partioned into two types, one further as Training1 and Testing1 set.

```r
Partition_Training<-createDataPartition(Training$classe,p=0.6,list=FALSE)
Training1<-Training[Partition_Training,]
dim(Training1)
```

```
## [1] 11776   160
```

```r
Testing1<-Training[-Partition_Training,]
dim(Testing1)
```

```
## [1] 7846  160
```
Cleaning the data can be done by using the NONZeroVariance function.By running this function in the new dataset Training1 will give you the dataset without NZV.


```r
#clearing NZV for Training1 dataset
Training1_NZV<-nearZeroVar(Training1,saveMetrics = TRUE)
Training1<-Training1[,Training1_NZV$nzv==FALSE]
dim(Training1)
```

```
## [1] 11776   131
```

```r
#clearing NZV for Testing1 dataset
Testing_NZV<-nearZeroVar(Testing1,saveMetrics = TRUE)
Testing1<-Testing1[,Testing_NZV$nzv==FALSE]
dim(Testing1)
```

```
## [1] 7846  132
```
Remove the First variable and clean variables with multiple NA.

```r
Training1<-Training1[c(-1)]
Training2<-Training1

for(i in 1:length(Training1))
{
        if(sum(is.na(Training1[,i]))/nrow(Training1)>=.6)
        {
                for(j in 1:length(Training2))
                {
                        if(length(grep(names(Training1[i]),names(Training2)[j]))==1)
                        {
                                Training2<-Training2[,-j]
                        }
                }
        }
}

Training1<-Training2

clean1 <- colnames(Training1)
##58 th column is classe and it will be removed
clean2 <- colnames(Training1[, -58]) 
Testing1 <- Testing1[clean1]          
Testing <- Testing[clean2]

dim(Testing1)
```

```
## [1] 7846   58
```
Coerce the data into same type is necessary during the use of various algorithms.


```r
for (i in 1:length(Testing) ) {
        for(j in 1:length(Training1)) {
        if( length( grep(names(Training1[i]), names(Testing)[j]) ) ==1)  {
            class(Testing[j]) <- class(Training1[i])
        }      
    }      
}
#And to make sure Coertion really worked, simple smart ass technique:
Testing <- rbind(Training1[2, -58] , Testing) #note row 2 does not mean anything, this will be removed right.. now:
Testing <- Testing[-1,]
dim(Testing)
```

```
## [1] 20 57
```

##Machine Learning Algorithms : Random Forest

```r
RandomSet <- randomForest(classe ~ ., data=Training1)
prediction <- predict(RandomSet, Testing1, type = "class")
ConfusionPrediction <- confusionMatrix(prediction, Testing1$classe)
ConfusionPrediction
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    3    0    0    0
##          B    1 1515    4    0    0
##          C    0    0 1364    0    0
##          D    0    0    0 1283    1
##          E    0    0    0    3 1441
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9985          
##                  95% CI : (0.9973, 0.9992)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9981          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9980   0.9971   0.9977   0.9993
## Specificity            0.9995   0.9992   1.0000   0.9998   0.9995
## Pos Pred Value         0.9987   0.9967   1.0000   0.9992   0.9979
## Neg Pred Value         0.9998   0.9995   0.9994   0.9995   0.9998
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1931   0.1738   0.1635   0.1837
## Detection Prevalence   0.2847   0.1937   0.1738   0.1637   0.1840
## Balanced Accuracy      0.9995   0.9986   0.9985   0.9988   0.9994
```
##Machine Learning Algorithm: Decision Tree

```r
DecisionTree <- rpart(classe ~ ., data=Training1, method="class")
#prp(DecisionTree)
rpart.plot(DecisionTree,main="TREE",extra = 102,under=TRUE,faclen = 0)
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-1.png) 

##Prediction Assignment Submission

```r
prediction <- predict(RandomSet, Testing, type = "class")
prediction
```

```
##  2  3  4  5 61  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

Function which will be used to submit the predictions for submission.

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(prediction)
```

