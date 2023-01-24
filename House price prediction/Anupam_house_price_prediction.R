#libraries
library(mlbench)
library(sf)
library(tmap)
library(caret)
library(gbm)
library(tidyverse)
library(GGally)
library(rgl)
library(reshape2)

?gbm()
?createDataPartition

data(BostonHousing)
head(BostonHousing)
BostonHousing[BostonHousing$rm>8,]
summary(BostonHousing$rm)
hist(BostonHousing$rm)
quantile(BostonHousing$rm)
length(which(BostonHousing$medv<25))
quantile(BostonHousing$zn)
quantile(BostonHousing$chas)
BostonHousing[1,14]
data_cor <- cor(BostonHousing[ , colnames(BostonHousing) != "medv"],  # Calculate correlations
                BostonHousing$medv)
data_cor 
set.seed(1234) #For reproducibility

#creating partition for training data
train.index = createDataPartition(BostonHousing$medv, p=0.7, list= F)

#dataframe of test and train data
data.train = BostonHousing[train.index,]
data.test = BostonHousing[-train.index,]

str()

#Summary of the target variable
summary(data.train$medv)
hist(data.train$medv)
summary(data.test$medv)
hist(data.test$medv)

#rescaling predictor variables
data.train.z =
  data.train %>% select(-medv) %>%
  mutate_if(is_logical,as.character) %>%
  mutate_if(is_double,scale) %>% data.frame()

data.test.z =
  data.test %>% select(-medv) %>%
  mutate_if(is_logical,as.character) %>%
  mutate_if(is_double,scale) %>% data.frame()

#add unscaled Y variable back
data.train.z$medv = data.train$medv
data.test.z$medv = data.test$medv

## Set up tuning grid
caretGrid <- expand.grid(interaction.depth=c(1, 3, 5,7,9), n.trees = (0:50)*50,
                         shrinkage=c(0.02, 0.03),
                         n.minobsinnode=2)

metric <- "RMSE"

trainControl <- trainControl(method="cv", number=10)

## run the model over the grid
set.seed(99)
gbm.caret <- train(medv ~ ., data = data.train.z, distribution="gaussian", method="gbm",
                   trControl=trainControl, verbose=FALSE,
                   tuneGrid=caretGrid, bag.fraction=0.75)

## Examine the results
print(gbm.caret)
ggplot(gbm.caret)

# explore the results
names(gbm.caret)

# see best tune
gbm.caret[6]

# see grid results
head(data.frame(gbm.caret[4]))

# check
dim(caretGrid)
dim(data.frame(gbm.caret[4]))

## Find the best parameter combination
# put into a data.frame
grid_df = data.frame(gbm.caret[4])
head(grid_df)

# confirm best model and assign to params object
grid_df[which.min(grid_df$results.RMSE), ]

# assign to params and inspect
params = grid_df[which.min(grid_df$results.RMSE), 1:4 ]
params

## Create final model
# because parameters are known, model can be fit without parameter tuning
fitControl <- trainControl(method = "none", classProbs = FALSE)

# extract the values from params
gbmFit <- train(medv ~ ., data=data.train.z, distribution="gaussian", method = "gbm",
                trControl = fitControl,
                verbose = FALSE,
                ## only a single model is passed to the
                tuneGrid = data.frame(interaction.depth = 5,
                                      n.trees = 850,
                                      shrinkage = 0.03,
                                      n.minobsinnode = 2))


## Prediction and Model evaluation
# generate predictions
pred = predict(gbmFit, newdata = data.test.z)
# plot these against observed
data.frame(Predicted = pred, Observed = data.test.z$medv) %>%
  ggplot(aes(x = Observed, y = Predicted))+ geom_point(size = 1, alpha = 0.5)+
  geom_smooth(method = "loess", col = "red")+
  geom_smooth(method = "lm")


# generate some prediction accuracy measures
postResample(pred = pred, obs = data.test.z$medv)
?ggplot

# examine variable importance
ggplot(varImp(gbmFit, scale = FALSE),colour='blue')

varImp(gbmFit, scale = FALSE)

print.varImp.13 <- function(x = vimp, top = 13) {
  printObj <- data.frame(as.matrix(sortImp(x, top)))
  printObj$name = rownames(printObj)
  printObj
}

df = data.frame(print.varImp.13(varImp(gbmFit)), method = "GBM")

df %>%
  ggplot(aes(reorder(name, Overall), Overall)) +
  geom_col(fill = "dark blue") +
  facet_wrap( ~ method, ncol = 3, scales = "fixed") +
  coord_flip() + xlab("") + ylab("Variable Importance") +
  theme(axis.text.y = element_text(size = 7)) + geom_text(aes(y=round(df$Overall,2), label=round(df$Overall,2)),size = 3.3, fontface = "bold", family = "Fira Sans"
  ) +
  #scale_x_continuous(expand = c(.01, .01)) +
  scale_fill_identity(guide = "none") +
  theme_void() +
  theme(
    axis.text.y = element_text(size = 14, hjust = 1, family = "Fira Sans"),
    plot.margin = margin(rep(15, 4))
  )
df %>%
  ggplot(aes(reorder(name, Overall), Overall)) +
  geom_col(fill = "dark blue") +
  facet_wrap( ~ method, ncol = 3, scales = "fixed") +
  coord_flip() + xlab("") + ylab("Variable Importance") +
  theme(axis.text.y = element_text(size = 10)) + geom_text(aes(y=round(df$Overall,2), label=round(df$Overall,2)),position = position_dodge(width = 1),
                                                           vjust = 0, hjust=-0.3,size = 6,fontface='bold'
  ) +
  theme(
    axis.text.y = element_text(size = 17, hjust = 1, family = "Fira Sans"),
    plot.margin = margin(rep(15, 4))
  )

?geom_text()

