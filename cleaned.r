#install.packages("tm")
#install.packages("SnowballC")
#install.packages("ROCR")
#install.packages("caret")
#install.packages("reldist")


library("tm")
library("SnowballC")
library("ROCR")
library("caret")
library("reldist")

# setwd("/home/patrick/Desktop/freelancePjs/spamClassifier/")

#read training files for spam training emails
# a  = Corpus(DirSource("D:/University/CMPE 428 - Data Science/Term Project/enron/spam/train"), readerControl = list(language="lat"))
a  = Corpus(DirSource("enron/spam/train"), readerControl = list(language="lat"))
#read training files for legitimate training emails
#b  = Corpus(DirSource("D:/University/CMPE 428 - Data Science/Term Project/enron/legitimate/train"), readerControl = list(language="lat"))
b  = Corpus(DirSource("enron/legitimate/train"), readerControl = list(language="lat"))
#read training files for spam test emails
#c  = Corpus(DirSource("D:/University/CMPE 428 - Data Science/Term Project/enron/spam/test"), readerControl = list(language="lat"))
c  = Corpus(DirSource("enron/spam/test"), readerControl = list(language="lat"))
#read training files for legitimate test emails
#d  = Corpus(DirSource("D:/University/CMPE 428 - Data Science/Term Project/enron/legitimate/test"), readerControl = list(language="lat"))
d  = Corpus(DirSource("enron/legitimate/test"), readerControl = list(language="lat"))

a = tm:::c.VCorpus(a,b)
a <- tm_map(a, removeNumbers)
a <- tm_map(a, removePunctuation)
a <- tm_map(a , stripWhitespace)
a <- tm_map(a, PlainTextDocument)
a <- tm_map(a, removeWords, stopwords("english")) # this stopword file is at C:\Users\[username]\Documents\R\win-library\2.13\tm\stopwords 
a <- tm_map(a, stemDocument, language = "english")

#discarding sparse terms
adtm <-DocumentTermMatrix(a) 

c = tm:::c.VCorpus(c,d)
c <- tm_map(c, removeNumbers)
c <- tm_map(c, removePunctuation)
c <- tm_map(c , stripWhitespace)
c <- tm_map(c, PlainTextDocument)
c <- tm_map(c, removeWords, stopwords("english")) # this stopword file is at C:\Users\[username]\Documents\R\win-library\2.13\tm\stopwords 
c <- tm_map(c, stemDocument, language = "english")

#discarding sparse terms
cdtm <-DocumentTermMatrix(c) 


# Train and Test Data

trainData <- as.data.frame(as.matrix(adtm))
colnames(trainData) <- make.names(colnames(trainData), unique = T)
rownames(trainData) <- as.character(1:2586)
trainData$response = factor(c(rep(1,750),rep(0,1836)), levels = c(0,1), labels=c("Negative","Positive"))

testData = as.data.frame(as.matrix(cdtm))
colnames(testData) <- make.names(colnames(testData), unique = T)
rownames(testData) <- as.character(1:2585)
testData$response = factor(c(rep(1,749),rep(0,1836)), levels = c(0,1), labels=c("Negative","Positive"))
testData = testData[sample(nrow(testData)),]

#term selection
maxFirst = 2586
maxSecond = 19552
chunks = trainData[1:maxFirst,1:maxSecond]
result = as.data.frame(t(chunks))



# freeup memory
rm(list = c("adtm","cdtm","chunks","a","b","c","d"))
detach("package:tm", unload=TRUE)



result$A = 0
result$B = 750
result$C = 0
result$D = 1836
result$ChiSqare = 0
result$OddsRatio = 0
result$GiniIndex = 0
a = 0
b = 0
c = 0
d = 0

s1 = apply(result, 1, FUN = function(x){return(sum(x[1:750]>0))})
s2 = apply(result, 1, FUN = function(x){return(sum(x[751:maxFirst]>0))})

result$A = s1
result$B = maxSecond - s1
result$C = s2
result$D = maxSecond - s2

a = result$A
b = result$B
c = result$C
d = result$D

result$ChiSqare = (maxFirst*((a*d - b*c))^2)/((a+b)*(a+c)*(b+d)*(c+d))
result$OddsRatio = (a*d) /(max(1,b)*max(1,c))
result$GiniIndex = (1/(a+c)^2)*(((a^2)/(a+b))^2+((c^2)/(c+d))^2)    

test = result$GiniIndex

gini(result[,2000])

chiOrdered = result[order(-result$ChiSqare),]
ChiValues = chiOrdered[1:1000,,]
ChiValues$RF = 0
ChiValues$OR = 0

ChiValues[,2594]= apply(ChiValues,1, FUN = function(x){
  a = x[2587]
  b = x[2588]
  c = x[2589]
  d = x[2590]
  return(log(2+(a/(max(1,c))),2))
})


ChiValues[,2595]= apply(ChiValues,1, FUN = function(x){
  a = x[2587]
  b = x[2588]
  c = x[2589]
  d = x[2590]
  return(log(2+(a*d)/(max(1,b)*max(1,c)),2))
})

chiVals = as.data.frame(t(ChiValues))
normalizedChi = chiVals
denom = 0
denom = apply(normalizedChi, 1, FUN = function(x){
  return(sqrt(sum(x^2)))
})


  
# I will use apply here as I always get confused with matrix vector division :(
normalizedChi = apply(normalizedChi,2,FUN = function(x){
  return(x / denom)
})


weightedChi = normalizedChi


# l = apply(weightedChi,1 , FUN = function(x){
#       return(x*x[2594])
#   })

#weightedChi = t(l)
chiResult = as.data.frame(weightedChi)

chiResult$response = factor(c(rep(1,750),rep(0,1845)), levels = c(0,1), labels=c("Negative","Positive"))


chiResult = chiResult[1:maxFirst,1:1001]


chiResult = chiResult[1:maxFirst,1:1001]
trainData = chiResult
trainData = trainData[sample(nrow(trainData)),]
#------------------------------------------------------- 
# running models 
#----------------------------------------
# editing the test dataframe to include the same column names as the training set
newtestData = testData[,which(colnames(testData) %in% colnames(trainData))]

other_word_not_in_test_data = trainData[-2586,-which((colnames(trainData) %in% colnames(testData)))]*0 

newtestData = cbind(newtestData,other_word_not_in_test_data)


#----------------------------------
library("rpart.plot")
tree1 = rpart(response~., data = trainData )
pdf("plot.pdf")
rpart.plot(tree1,roundint = FALSE)
dev.off()

resultDecision <- predict(tree1, newdata=newtestData, type="class")

library(e1071)
library(caret)
con = confusionMatrix(resultDecision, newtestData$response)
precision <- con$byClass['Pos Pred Value']    
recall <- con$byClass['Sensitivity']
f1 <- 2 * ((precision * recall) / (precision + recall))

#-------------------------------------------
logModel = glm(response~.,family = binomial(link = "logit"),data = trainData)

resultLog = predict(logModel,newdata = newtestData,type="response") 
resultLog <- ifelse(resultLog > .5, "Positive","Negative")
resultLog = factor(resultLog)

con = confusionMatrix(resultLog, newtestData$response)
precision <- con$byClass['Pos Pred Value']    
recall <- con$byClass['Sensitivity']
f1 <- 2 * ((precision * recall) / (precision + recall))

#----------------------------------------------------
# SVM
#----------------------------------------------------
library(caret)
library(kernlab)
library(ROCR)


#trainLabel <- as.factor(trainData$response)
#testLabel  <- as.factor(testData$response)

trainData$response <- as.factor(trainData$response)
testData$response <- as.factor(testData$response)


modelParams <- trainControl(method="none",
                           verboseIter=FALSE,
                           classProbs=TRUE,
                           summaryFunction = twoClassSummary)  

modelFit <- train(response ~., data = trainData,
            method = "svmLinear",
            preProc = c("center","scale"),
            trControl=modelParams,
            metric = "ROC")


predProb    = predict(modelFit,newdata = newtestData,type="raw") 



predClass   = predict(modelFit,newdata = newtestData,type="raw") 


con = confusionMatrix(predClass, newtestData$response)
precision <- con$byClass['Pos Pred Value']    
recall <- con$byClass['Sensitivity']
f1 <- 2 * ((precision * recall) / (precision + recall))
