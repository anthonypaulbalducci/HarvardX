# Presumes the required RTools 4.3 >= is already installed.
# See: https://cran.r-project.org/bin/windows/Rtools/rtools43/rtools.html

 if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
 if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
 if(!require(caretEnsemble)) install.packages("caretEnsemble", repos = "http://cran.us.r-project.org")
 if(!require(recommenderlab)) install.packages("recommenderlab", repos = "http://cran.us.r-project.org")
 if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
 if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
 if(!require(Metrics)) install.packages("Metrics", repos = "http://cran.us.r-project.org")
 if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
 if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
 if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
 if(!require(doParallel)) install.packages("doParallel", repos = "http://cran.us.r-project.org")
 if(!require(parallel)) install.packages("parallel", repos = "http://cran.us.r-project.org")
 

library(tidyverse)
library(caret)
library(caretEnsemble) # Library allows easy generation of ensembles with caret
library(ggplot2)
library(dplyr)
library(Metrics)
library(lubridate)
library(glmnet) 
library(recosystem) # Recommender systems package with Matrix Factorization capabilities
library(doParallel) # Both parallel and doParallel are required for parallel processing
library(parallel)   # on Windows.
library(scales)


# Note: Configures parallel processing with the 'doParallel'
# package. Actual setting will depend on user's configuration.
# For reference I am performing analysis on a 6-core (12 logical) AMD Ryzen 5 5600X with
# 96 GB RAM and 2x 512 GB SSD storage in a RAID-0 ('striping') configuration.
# R Version: 4.3.2

numCores<-detectCores() # 12 cores detected

# Overloading the system can produce a 'Error in unserialize(socklist[[n]]) : error reading from connection' condition
# In practice I found leaving at least 20-30% CPU/RAM resources at minimum is required
# to produce stable results.

pCluster <- makeCluster(numCores[1]-4) # Leaves four cores free for required concurrent processes.

# Register cluster

registerDoParallel(pCluster)

# Edx supplied boiler-plate data download and data set assembly code.
# Easily 'uncomment' in RStudio by making selection and then (on PC) via
# 'Ctrl-Shift-C' if needed.

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

########################################
# Data Wrangling / Feature Engineering #
########################################

edx <- mutate(edx, date = as_date(as_datetime(timestamp))) # Convert timestamp to date.
edx <- mutate(edx, date_cuts = cut(edx$date, breaks="40 days", labels=FALSE))
# Converts dates to breaks in 40 day increments to account for delay in
# 'change of opinion' time effects.
# Adds an additional feature for consideration.


####################
# Data Exploration #
####################

# Before formally entering into exploration of the data set let's start by ensuring
# we don't have any NA/NULL entries in either our train or holdout test to deal with
# that need to be cleaned. The presence of such entries can lead to a failed analysis
# or other issues such multicollinearity that will have to be addressed before
# continuing.

sum(is.na(edx)) # Sums to '0'-- No NA's present in train set.
sum(is.na(final_holdout_test)) # Sums to '0'-- No NA's present in test set.

numReviewsByUser <- table(edx$userId) %>% as.data.frame() %>% 
  arrange(Freq) # Find, sort, and plot the frequency of user reviews by userId ascending.

colnames(numReviewsByUser)[1] <- "userId" # Set column header

meanNumReviewsByUser <- mean(numReviewsByUser$Freq)                        ############################
medianNumReviewsByUser <- median(numReviewsByUser$Freq)                    # Common exploratory stats #
modeNumReviewsByUser <- mode(numReviewsByUser$Freq)                        ############################
maxNoReviews <- max(numReviewsByUser$Freq)
minNoReviews <- min(numReviewsByUser$Freq)
quantileNumReviewsByUser <- quantile(numReviewsByUser$Freq, probs = c(0,0.25,0.5,0.75,1))

# Note: Majority of previous stats can be obtained more easily simply using summary(numReviewsByUser$Freq)
# and then running str() on the result (i.e. stats <- summary(numReviewsByUser$Freq), and then realizing 
# the result is an atomic vector they can be pulled out directly. For example stats[4] will give you the 
# mean. However I felt listing them out was more readable for the lay user).

ggplot(numReviewsByUser, aes(x = as.numeric(row.names(numReviewsByUser)), y=Freq)) + 
  geom_point(aes(color=Freq)) + scale_color_gradient(low="blue", high="red") + 
  labs(x = "User order by rank in number of reviews", y = "Number of reviews", color = "Frequency", title = "Number of reviews in MovieLens database ranked by user") + 
  theme(plot.title = element_text(hjust = 0.5)) + geom_line(aes(y = meanNumReviewsByUser)) + 
  annotate("text", x=17000, y=310, label="Mean Reviews (128.7967)") # Note: There are a lot of data points here, so the chart make take a little to load

ggplot(edx, aes(x = date)) + geom_histogram(colour = 4, fill = "white", bins = 14) + scale_x_date(limits = as.Date(c("1995-01-09", "2009-01-05")), breaks = 14 ) + 
  scale_y_continuous(labels = comma, breaks = c(250000, 500000, 750000, 1000000, 1250000)) + 
  scale_x_date(date_breaks = "1 years", date_labels = "%Y", limits = as.Date(c("1995-01-09", "2009-01-05"))) + 
  labs(x = "Year", y = "Number of Reviews", title = "Number of reviews by year in EDX MovieLens data") + 
  theme(plot.title = element_text(hjust = 0.5))

############
# Analysis #
############

set.seed(123, sample.kind="Rounding")

# Create edx train/test set
# Seeing as it is bad practice to do model selection on
# our final hold-out test, we create our own sub train /
# test sets on the edx data to play with. Further, in 
# practice, we might not even have a final prediction /  
# validation set available to us at the time of model  
# construction.

test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)

edx_train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in edx test set are also in edx train set
edx_test_set <- temp %>% 
  semi_join(edx_train_set, by = "movieId") %>%
  semi_join(edx_train_set, by = "userId")

rm(test_index, temp)

####################
#User Effects Model#
####################

# Regularization
# Elastic Net Regression - 

train_set_sample <- slice_sample(edx_train_set, n = 30000, replace = TRUE)

# Provided such a small sample size is being used and the data are factors
# *not* continuous we must again trim our test set by the userId and
# movieId present in the train_set_sample, or undoubtedly the end result of
# the prediction will fail.

edx_test_set <- edx_test_set %>% 
  semi_join(train_set_sample, by = "movieId") %>%
  semi_join(train_set_sample, by = "userId")

# Set training control
train_control <- trainControl(method = "repeatedcv", # 5x reoeated CV run once
                              number = 5,
                              repeats = 1,
                              search = "random",
                              verboseIter = TRUE,
                              allowParallel = TRUE)

print("Commencing runtime analysis")
start_time <- Sys.time() # Run-time analysis start

# Train the model
elastic_net_model <- train(rating ~ as.factor(userId),
                           data = train_set_sample,
                           method = "glmnet",
                           preProcess = c("center", "scale"),
                           tuneLength = 10,
                           trControl = train_control,
                           na.action = na.pass)
end_time <- Sys.time() # Run-time analysis end

print(paste("Elapsed Analysis Runtime:", end_time - start_time)) # Run-time of model analysis displayed

# Check multiple R-squared
result <- predict(elastic_net_model, edx_test_set)
RE_U <- RMSE(result, edx_test_set$rating)
print(paste("RMSE - User Effects:", RE_U))

##############################
# User + Movie effects Model #
############################## 

# Set training control
train_control <- trainControl(method = "repeatedcv", # 5x reoeated CV run once
                              number = 5,
                              repeats = 1,
                              search = "random",
                              verboseIter = TRUE,
                              allowParallel = TRUE)

print("Commencing runtime analysis")
start_time <- Sys.time() # Run-time analysis start

# Train the model
elastic_net_model <- train(rating ~ as.factor(movieId) + as.factor(userId),
                           data = train_set_sample,
                           method = "glmnet",
                           preProcess = c("center", "scale"),
                           tuneLength = 10,
                           trControl = train_control,
                           na.action = na.pass)
end_time <- Sys.time() # Run-time analysis end

print(paste("Elapsed Analysis Runtime:", end_time - start_time)) # Run-time of model analysis displayed

# Check multiple R-squared
result <- predict(elastic_net_model, edx_test_set)
RE_UM <- RMSE(result, edx_test_set$rating)
print(paste("RMSE - User + Movie Effects:",RE_UM))

###############################
# User + Movie + Time Effects #
###############################

# Set training control
train_control <- trainControl(method = "repeatedcv", # 5x reoeated CV run once
                              number = 5,
                              repeats = 1,
                              search = "random",
                              verboseIter = TRUE,
                              allowParallel = TRUE)

print("Commencing runtime analysis")
start_time <- Sys.time() # Run-time analysis start

# Train the model
elastic_net_model <- train(rating ~ as.factor(movieId) + as.factor(userId) + as.factor(date_cuts),
                           data = train_set_sample,
                           method = "glmnet",
                           preProcess = c("center", "scale"),
                           tuneLength = 10,
                           trControl = train_control,
                           na.action = na.pass)
end_time <- Sys.time() # Run-time analysis end
print(paste("Elapsed Analysis Runtime:", end_time - start_time)) # Run-time of model analysis displayed

# Check multiple R-squared
result <- predict(elastic_net_model, edx_test_set)
RE <- RMSE(result, edx_test_set$rating)
print(RE)

###################################
#Recosystem - Matrix Factorization#
###################################

# Trained on the *entire* EDX data set, which is simply not possible with the regression
# methods used above, due to both time and resources.

ML_train_set <-  with(edx, data_memory(user_index = userId, 
                                       item_index = movieId, 
                                       rating     = rating))

ML_test_set  <-  with(final_holdout_test,  data_memory(user_index = userId, 
                                                       item_index = movieId, 
                                                       rating     = rating))
MovieLensData = Reco()

opts = MovieLensData$tune(ML_train_set, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.25),
                                                    costp_l1 = 0, costq_l1 = 0,
                                                    nthread = 6, niter = 10, verbose = TRUE))

MovieLensData$train(ML_train_set, opts = c(opts$min, nthread = 6, niter = 30))

result <-  MovieLensData$predict(ML_test_set, out_memory())

RE_reco <- RMSE(result, edx_test_set$rating)
print(RE_reco)

final_error_measure <- RMSE(result, final_holdout_test$rating)
print(final_error_measure)



















edx_train_set_sample <- slice_sample(edx_train_set, n = 40000) # Generate random sample from train set for analysis

print("Commencing runtime analysis")
start_time <- Sys.time() # Run-time analysis start

trainControl (verboseIter = TRUE, allowParallel = TRUE, number = 5, method = "cv") 
fit <- train(rating ~ as.factor(userId), method = "lm", data = edx_train_set_sample, trControl = trainControl())
end_time <- Sys.time() # Run-time analysis end
print(paste("Elapsed Analysis Runtime:", end_time - start_time)) # Run-time of model analysis displayed

results <- predict(fit, edx_test_set)
error_measure <- RMSE(results, edx_test_set$rating)

error_measure

print("Commencing runtime analysis")
start_time <- Sys.time() # Run-time analysis start

trainControl (verboseIter = TRUE, allowParallel = TRUE, number = 5, method = "cv") 
fit <- train(rating ~ as.factor(userId) + as.factor(movieId), method = "lm", data = edx_train_set, trControl = trainControl())
end_time <- Sys.time() # Run-time analysis end
print(paste("Elapsed Analysis Runtime:", end_time - start_time)) # Run-time of model analysis displayed

results <- predict(fit, edx_test_set)
error_measure <- RMSE(results, edx_test_set$rating)



