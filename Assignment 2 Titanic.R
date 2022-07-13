library(ggplot2)
library(dplyr)
library(Amelia)
library(randomForest)

# Reading dataframe and converting empty values to NA values
titanic.df <- read.csv("Titanic train.csv", na.strings = "")

View(titanic.df)

summary(titanic.df)
str(titanic.df)

#Pre-processing

library(Amelia)
missmap(titanic.df, col=c("orange", "black"))

# We can observe that cabin has many missing values. It's best to drop it
# Assumption: Passenger ID, Name, Fare, Embarked Port and Ticket columns will not determine the survival
# It is decided to keep 6 variables: Survived, Pclass, Age, Sex, SibSp, Parch

library(dplyr)
titanic.df <- select(titanic.df, Survived, Pclass, Age, Sex, SibSp, Parch)

any(is.na(titanic.df))

titanic.df = na.omit(titanic.df)

View(titanic.df)

# Checking that all NA values have been managed
missmap(titanic.df, col=c("orange", "black"))

str(titanic.df)

# Converting Survived and Pclass into factors, not int
titanic.df$Survived = factor(titanic.df$Survived)
titanic.df$Pclass = factor(titanic.df$Pclass)

str(titanic.df)


# Exploratory analisys : plotting graphics

  #Survival count

plot1 <- (ggplot(titanic.df, aes(x = Survived)))+  geom_bar(color = 'blue', fill = 'orange', size = 1)
print (plot1 + xlab('Survived') + ggtitle('Survival count'))
  
  # Class count
plot2 <- ggplot(titanic.df, aes(x = Pclass))
print(plot2 + geom_bar(color = 'blue', fill = 'green', size = 1)
      + xlab('Class')+ ggtitle('Class'))

  # Sex count
ggplot(titanic.df,aes(Sex)) + geom_bar(aes(fill=factor(Sex)))

    # Survive count by sex
ggplot(titanic.df, aes(x = Survived, fill=Sex)) +
  geom_bar(position = position_dodge()) +
  geom_text(stat='count', aes(label=stat(count)), 
            position = position_dodge(width=1), vjust=-0.5)
  

# Survive count by class

ggplot(titanic.df, aes(x = Survived, fill=Pclass)) +
  geom_bar(position = position_dodge()) +
  geom_text(stat='count', 
      aes(label=stat(count)), 
      position = position_dodge(width=1), vjust=-0.5)


  #age distribution
ggplot(titanic.df,aes(Age)) + geom_histogram(fill='blue',bins=20,alpha=0.5)



# Splitting the dataset into train and test with a 70/30 ratio

set.seed(5)
index <- sample(nrow(titanic.df), 0.7 * nrow(titanic.df)) 
titanic.train <- titanic.df[index,] 
titanic.test <- titanic.df[-index,]

str(titanic.train)

str(titanic.test)





# Random Forests

  # Perform training:

library(randomForest)
random.forest.training <- randomForest(Survived ~ ., data=titanic.train)

  # Running the prediction 
rf.prediction <- predict(random.forest.training, titanic.test[,-1])

  # Random forest confusion matrix
rf.confumat <- table(observed=titanic.test[,1],predicted=rf.prediction)
print(rf.confumat)

  # Misclassification error 

rf.misClassError <- mean(rf.prediction != factor(titanic.test$Survived))
print(rf.misClassError)

  # RF: calculating accuracy, precision sensitivity, fscore and specificity
rf.accuracy <- sum(rf.confumat[1], rf.confumat[4]) / sum(rf.confumat[1:4])
rf.precision <- rf.confumat[4] / sum(rf.confumat[4], rf.confumat[2])
rf.sensitivity <- rf.confumat[4] / sum(rf.confumat[4], rf.confumat[3])
rf.fscore <- (2 * (rf.sensitivity * rf.precision))/(rf.sensitivity + rf.precision)
rf.specificity <- rf.confumat[1] / sum(rf.confumat[1], rf.confumat[2])



# Logistic regression

logistic.model <- glm(formula = Survived ~ . ,family=binomial(link = 'logit'), data = titanic.train)

  # From summary we can  see how Class, Age and Sex are the most significant features
summary(logistic.model)

  # running the prediction
lm.prediction <- predict(logistic.model, newdata = titanic.test, type = 'response')
lm.results <- ifelse (lm.prediction >0.5, 1, 0)

  # Logistic regression confusion matrix
lm.confumat <- table(observed=titanic.test$Survived, predicted=lm.results > 0.5)
print(lm.confumat)

  # missclassification error and accuracy
lm.missClassError <- mean(lm.results != factor(titanic.test$Survived))
print(lm.missClassError)


  # LM: calculating accuracy, precision sensitivity, fscore and specificity
lm.accuracy <- sum(lm.confumat[1], lm.confumat[4]) / sum(lm.confumat[1:4])
lm.precision <- lm.confumat[4] / sum(lm.confumat[4], lm.confumat[2])
lm.sensitivity <- lm.confumat[4] / sum(lm.confumat[4], lm.confumat[3])
lm.fscore <- (2 * (lm.sensitivity * lm.precision))/(lm.sensitivity + lm.precision)
lm.specificity <- lm.confumat[1] / sum(lm.confumat[1], lm.confumat[2])


  # Creating comparison table 
results.table <- matrix(c(rf.accuracy, rf.precision, rf.sensitivity, rf.fscore, rf.specificity,
                          lm.accuracy, lm.precision, lm.sensitivity, lm.fscore, lm.specificity),
                    ncol = 5, byrow = TRUE)
rownames(results.table) <- c("Random Forest", "Logistic regression")
colnames(results.table) <- c("Accuracy", "Precision", "Sensitivity", "F-score", "Specificity")
results.table <- as.table(results.table)

results.table