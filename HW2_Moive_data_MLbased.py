# Databricks notebook source
# MAGIC %md 
# MAGIC ### Spark Moive Recommendation
# MAGIC In this notebook, we will use an Alternating Least Squares (ALS) algorithm with Spark APIs to predict the ratings for the movies in [MovieLens small dataset](https://grouplens.org/datasets/movielens/latest/)

# COMMAND ----------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# COMMAND ----------

import os
os.environ["PYSPARK_PYTHON"] = "python3"

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part0: Data Import

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("moive analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# COMMAND ----------

movies_df = spark.read.load("/FileStore/tables/movies.csv", format='csv', header = True)
ratings_df = spark.read.load("/FileStore/tables/ratings.csv", format='csv', header = True)
links_df = spark.read.load("/FileStore/tables/links.csv", format='csv', header = True)
tags_df = spark.read.load("/FileStore/tables/tags.csv", format='csv', header = True)

# COMMAND ----------

movies_df.show(5)

# COMMAND ----------

ratings_df.show(5)

# COMMAND ----------

links_df.show(5)

# COMMAND ----------

tags_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part I: Exploratory Data Analysis

# COMMAND ----------

tmp1 = ratings_df.groupBy("userID").count().toPandas()['count'].min()
tmp2 = ratings_df.groupBy("movieId").count().toPandas()['count'].min()
print('For the users that rated movies and the movies that were rated:')
print('Minimum number of ratings per user is {}'.format(tmp1))
print('Minimum number of ratings per movie is {}'.format(tmp2))

# COMMAND ----------

tmp1 = sum(ratings_df.groupBy("movieId").count().toPandas()['count'] == 1)
tmp2 = ratings_df.select('movieId').distinct().count()
print('{} out of {} movies are rated by only one user'.format(tmp1, tmp2))

# COMMAND ----------

tmp1 = ratings_df.groupBy("movieId").count().toPandas()['count'].mean()
tmp2 = ratings_df.groupBy("userId").count().toPandas()['count'].mean()
print('Average number of ratings per movie is {}'.format(tmp1))
print('Average number of ratings per user is {}'.format(tmp2))

# COMMAND ----------

movies_df.registerTempTable("movies")
ratings_df.registerTempTable("ratings")
links_df.registerTempTable("links")
tags_df.registerTempTable("tags")

# COMMAND ----------

# MAGIC %md ### The number of Users

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT Count(DISTINCT userId) AS number_of_users 
# MAGIC FROM ratings

# COMMAND ----------

# MAGIC %md ###The number of Movies

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT Count(DISTINCT movieID) AS number_of_movies
# MAGIC FROM movies

# COMMAND ----------

# MAGIC %md
# MAGIC ### The sparsity of the movie ratings
# MAGIC 
# MAGIC ### sparcity = 1- rating_num/(movie_num*user_num)

# COMMAND ----------

movie_num = ratings_df.select('movieId').distinct().count()
user_num = ratings_df.select('userId').distinct().count()
rating_num = ratings_df.select('rating').count()

# COMMAND ----------

denominator = movie_num*user_num
numerator = rating_num
sparsity = (1-numerator/denominator)*100
print ("The ratings dataframe is ", "%.2f" % sparsity + "% empty.")

# COMMAND ----------

# MAGIC %md ### Movies rated by users. 

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT Count(DISTINCT movieID) AS movies_rated
# MAGIC FROM ratings 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Movies not rated before.

# COMMAND ----------

spark.sql(
  '''
SELECT m.*,r.rating
FROM movies m 
    LEFT JOIN ratings r ON m.movieId = r.movieId
WHERE r.rating is Null
'''
).show(5)

# COMMAND ----------

# MAGIC %md ### List Movie Genres

# COMMAND ----------

# MAGIC %md
# MAGIC Each movie belongs to more than 1 genre,as shown below. We need to seperate these genres.

# COMMAND ----------

spark.sql(
  '''
SELECT DISTINCT title, genres
FROM movies
  '''
).show(5)

# COMMAND ----------

## Data processing to seperate the genres for a movie
genres_pd_df = spark.sql("SELECT DISTINCT title, genres FROM movies").toPandas()
genres_pd_df['genres'] = genres_pd_df['genres'].apply(lambda x:x.split('|'))
genres_pd_df = pd.concat([genres_pd_df['title'],genres_pd_df['genres'].apply(pd.Series)],axis = 1).set_index('title')
genres_sep_pd_df = genres_pd_df.stack().reset_index(level=0)
genres_sep_pd_df.columns = ['title','genre']

# COMMAND ----------

# MAGIC %md
# MAGIC We store all the movie genres in a list called movie_genres_list.

# COMMAND ----------

movie_genres_list = genres_sep_pd_df['genre'].unique().tolist()
print('Here are all the movie genres:'+'\n', movie_genres_list)

# COMMAND ----------

# MAGIC %md ### Movie for Each Category

# COMMAND ----------

genres_sep_df = sqlContext.createDataFrame(genres_sep_pd_df)
genres_sep_df.registerTempTable('genres_sep')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT genre,count(*) AS count
# MAGIC FROM genres_sep
# MAGIC GROUP BY 1
# MAGIC ORDER BY 2 DESC

# COMMAND ----------

# MAGIC %md
# MAGIC We put each movie under its genre. This is stored in a dictionary called movie_genre_dict, where the key is the genre, and the values is a list that contains the movies belonging to this genre.

# COMMAND ----------

movie_genre_dict = dict()
for _ in movie_genres_list:
  movie_genre_dict[_] = list()
for index, row in genres_sep_pd_df.iterrows():
   movie_genre_dict[row["genre"]].append(row["title"])

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at the crime genre as an example.

# COMMAND ----------

print('Here are movies that belong to the crime genre:'+'\n',movie_genre_dict['Crime'])

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part2: Spark ALS based approach for training model
# MAGIC We will use an Spark ML to predict the ratings, so let's reload "ratings.csv" using ``sc.textFile`` and then convert it to the form of (user, item, rating) tuples.

# COMMAND ----------

ratings_df.show()

# COMMAND ----------

movie_ratings_df=ratings_df.drop('timestamp')
movie_ratings_df.show()

# COMMAND ----------

# Data type convert
from pyspark.sql.types import IntegerType, FloatType
movie_ratings_df = movie_ratings_df.withColumn("userId", movie_ratings_df["userId"].cast(IntegerType()))
movie_ratings_df = movie_ratings_df.withColumn("movieId", movie_ratings_df["movieId"].cast(IntegerType()))
movie_ratings_df = movie_ratings_df.withColumn("rating", movie_ratings_df["rating"].cast(FloatType()))

# COMMAND ----------

movie_ratings_df.show(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### ALS Model Selection and Evaluation
# MAGIC 
# MAGIC With the ALS model, we can use a grid search to find the optimal hyperparameters.

# COMMAND ----------

# import package
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder

# COMMAND ----------

#Create test and train set
(training,test)=movie_ratings_df.randomSplit([0.8,0.2],seed = 42)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## We will tune the hyperparameters using ParamGridBuilder and CrossValidator.

# COMMAND ----------

#Create ALS model
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",coldStartStrategy = 'drop',nonnegative = True, implicitPrefs = False)
# Confirm that a model called "als" was created
type(als)

# COMMAND ----------

#Tune model using ParamGridBuilder
# We will just tune rank and regParam considering long run time, after we get the have combination, we will use larger iterations.
param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [3,5,10]) \
            .addGrid(als.maxIter, [10]) \
            .addGrid(als.regParam, [0.05,0.15,0.25]) \
            .build()
print ("Num models to be tested: ", len(param_grid))

# COMMAND ----------

# Define evaluator as RMSE
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction") 

# COMMAND ----------

# Build Cross validation 
cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)

# COMMAND ----------

#Fit ALS model to training data
model = cv.fit(training)

# COMMAND ----------

#Extract best model from the tuning exercise using ParamGridBuilder
best_model = model.bestModel

# COMMAND ----------

#Generate predictions and evaluate using RMSE
predictions=best_model.transform(test)
rmse = evaluator.evaluate(predictions)

# COMMAND ----------

#Print evaluation metrics and model parameters
print ("RMSE = "+str(rmse))
print ("**Best Model**")
print (" Rank:",best_model._java_obj.parent().getRank())   #parent()method will return an estimator,you can get the best params then
print (" MaxIter:",best_model._java_obj.parent().getMaxIter())
print (" RegParam:",best_model._java_obj.parent().getRegParam()) 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Increase iteration number and Model testing

# COMMAND ----------

#Increase the interation for the best ALS model
#coldStartStrategy = 'drop' is important, otherwise, you will recieve rmse = nan
#Spark allows users to set the coldStartStrategy parameter to “drop” in order to drop any rows in the DataFrame of predictions that contain NaN values. 
als_50 = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",rank = 10, maxIter = 50,regParam = 0.15, nonnegative = True, coldStartStrategy = 'drop',implicitPrefs = False)
#fit the model to training data
best_model_50 = als_50.fit(training)

# COMMAND ----------

#generate predictions on test data
prediction_50 = best_model_50.transform(test)

# COMMAND ----------

#tell spark how to evaluate predictions
evaluator_50 = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
#obtain rmse
rmse_50 = evaluator_50.evaluate(prediction_50)
#print rmse
print('RMSE=',rmse_50)

# COMMAND ----------

prediction_50_pd_df = prediction_50.toPandas()[['movieId','rating','prediction']].set_index('movieId')
display(prediction_50_pd_df.plot(style=['o','rx']))

# COMMAND ----------

# round the prediction to a scale of 1-5
def round_to_5scale(x):
  if x<=round(x) and 0.5<=x<=5:
    return round(x)-0.5 if round(x)-x>0.25 else round(x)
  if x>round(x) and 0.5<=x<=5:
    return round(x)+0.5 if x-round(x)>0.25 else round(x)
  if x>5:
    return 5
  if x<0.5:
    return 0.5

# COMMAND ----------

prediction_50_pd_df['prediction'] = prediction_50_pd_df['prediction'].apply(lambda x: round_to_5scale(x))
display(prediction_50_pd_df.plot(style=['o','rx']))

# COMMAND ----------

prediction_50_user_pd_df = prediction_50.toPandas()[['userId','rating','prediction']].set_index('userId')
prediction_50_user_pd_df['prediction'] = prediction_50_user_pd_df['prediction'].apply(lambda x: round_to_5scale(x))
display(prediction_50_user_pd_df.plot(style=['o','rx']))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Apply model to all data and see the performance

# COMMAND ----------

#Best_model RMSE
alldata=best_model.transform(movie_ratings_df)
rmse = evaluator.evaluate(alldata)
print ("RMSE = "+str(rmse))

# COMMAND ----------

#Best_model_50 RMSE
alldata=best_model_50.transform(movie_ratings_df)
rmse = evaluator.evaluate(alldata)
print ("RMSE = "+str(rmse))

# COMMAND ----------

alldata.registerTempTable("alldata")

# COMMAND ----------

sparl.sql(
  '''
SELECT *
FROM movies
	JOIN alldata ON movies.movieId = alldata.movieId
  '''
).show(5)

# COMMAND ----------

alldata_pd_df = alldata.toPandas()[['movieId','rating','prediction']].set_index('movieId')
alldata_pd_df.plot(style=['o','rx'])
display()

# COMMAND ----------

alldata_pd_df['prediction'] = alldata_pd_df['prediction'].apply(lambda x: round_to_5scale(x))
alldata_pd_df.plot(style=['o','rx'])
display()

# COMMAND ----------

alldata_user_pd_df = alldata.toPandas()[['userId','rating','prediction']].set_index('userId')
alldata_user_pd_df['prediction'] = alldata_user_pd_df['prediction'].apply(lambda x: round_to_5scale(x))
display(alldata_user_pd_df.plot(style=['o','rx']))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's look more closely on the user and movie data to decide which user to recommend.

# COMMAND ----------

ratings_info_df = movie_ratings_df.groupBy('movieId').avg('rating')
movie_ratings_count= movie_ratings_df.groupBy('movieId').count()
ratings_info_df = ratings_info_df.join(movie_ratings_count,'movieId','left').join(movies_df,'movieId','left')
ratings_info_df.show(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### UserId order by rating count

# COMMAND ----------

ratings_df.groupBy("userId").count().toPandas().sort_values(by = 'count',ascending=False).head()

# COMMAND ----------

# MAGIC %md
# MAGIC We will recommend user 414 and 599 as they have the highest rating count. This means the recommendations to them would be more reliable.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Histogram of rating counts

# COMMAND ----------

display(ratings_info_df.toPandas()['count'].hist(bins=50,log= True))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Histogram of the ratings

# COMMAND ----------

display(ratings_info_df.toPandas()['avg(rating)'].hist(bins=50))

# COMMAND ----------

fig, ax = plt.subplots()
g = sns.jointplot(x='avg(rating)', y='count', data=ratings_info_df.toPandas())
display(g.fig)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Recommend moive to users with id: 414, 599.  

# COMMAND ----------

# use the recommendation function of ALS
ALS_recommendations = best_model.recommendForAllUsers(10)
ALS_recommendations.filter(ALS_recommendations['userId'] == 599).show()

# COMMAND ----------

# MAGIC %md
# MAGIC We need to process the above dataframe for readability.

# COMMAND ----------

# Data procesing of the ALS_recommendations dataframe
from pyspark.sql.functions import explode,col
recommendations_df = (ALS_recommendations\
                      .select("userId",\
                              explode("recommendations")\
                              .alias("recommendation"))\
                      .select("userId", "recommendation.movieId",\
                              col("recommendation.rating")\
                              .alias('prediction')))
recommendations_df.show(3)

# COMMAND ----------

# we only recommend movies that have not been watched by users before
recommendations_df = recommendations_df.join(movies_df,["movieId"],"left").join(ratings_df,['movieId','UserId'],'left')
recommendations_df = recommendations_df.drop('timestamp')
recommendations_df = recommendations_df.filter(ratings_df.rating.isNull())
recommendations_df.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC We will only recommend movies that the users haven't watched/rated.
# MAGIC 
# MAGIC movies recommendations for user 414

# COMMAND ----------

recommendations_df.filter(recommendations_df['userId'] == 414).show()

# COMMAND ----------

# MAGIC %md 
# MAGIC movies recommendations for user 599

# COMMAND ----------

recommendations_df.filter(recommendations_df['userId'] == 599).show()

# COMMAND ----------

# Another way is to recommendForUserSubset function
#users = ALS_recommendations.filter(ALS_recommendations['userId'] == 575)
#ALS_recommendations_target = best_model.recommendForUserSubset(users,1)
#ALS_recommendations_target.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find the similar moives for moive with id: 464, 471

# COMMAND ----------

# MAGIC %md
# MAGIC The similarites of different movies can be recognized from correlations of their user ratings. Let's say user A and B both gave movie 1,2 5 star rating. This indicates that the movie 1 and 2 might be highily similar.
# MAGIC 
# MAGIC However, we have a challenge in that some of the movies have very few ratings and may end up having high correlation simply because one or two people gave them a 5 star rating. We can fix this by setting a threshold for the number of ratings. From the histogram earlier we saw a sharp decline in number of ratings from 100. Therefore we will choose this as our threshold.

# COMMAND ----------

movie_matrix = movie_ratings_df.toPandas().pivot_table(index='userId', columns='movieId', values='rating')

# COMMAND ----------

def find_similar_movie(x):
  movie_x_rating = movie_matrix[x]
  similar_to_x=movie_matrix.corrwith(movie_x_rating).reset_index(level=0)
  similar_to_x.dropna(axis = 0,how = 'any',inplace=True)
  similar_to_x.columns = ['movieId','correlation']
  
  similar_to_x_df = sqlContext.createDataFrame(similar_to_x)
  similar_to_x_movie = similar_to_x_df.join(ratings_info_df,'movieId','left').toPandas()[['movieId','correlation','title','count']]
  res = similar_to_x_movie[similar_to_x_movie['count']>100].sort_values(by = 'correlation',ascending = False)
  return similar_to_x,res

# COMMAND ----------

# MAGIC %md 
# MAGIC We will only find similar movies that have been rated more than 100 times.
# MAGIC 
# MAGIC movies similar to movie 471

# COMMAND ----------

# Movies similar to 471
corr_471,similar_to_471_movie = find_similar_movie(471)

# COMMAND ----------

corr_471.plot(kind='scatter',x = 'movieId',y='correlation')
display()

# COMMAND ----------

similar_to_471_movie.head()

# COMMAND ----------

# MAGIC %md 
# MAGIC movies similar to movie 464

# COMMAND ----------

# Movies similar to 464
corr_464,similar_to_464_movie = find_similar_movie(464)

# COMMAND ----------

corr_464.plot(kind='scatter',x = 'movieId',y='correlation')
display()

# COMMAND ----------

similar_to_464_movie.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Report
# MAGIC In this project, the movie data from movie lens data set which includes about 600 users and 9500 movies were analyzed to gain insights on movie recomendations to users and finding similar movies. At the beginning, we calculate the sparisity of the movie ratings which is 98.3%, this tells us that the ratings dataframe is mostly empty, which brings significance to predict the user ratings from what we have.
# MAGIC 
# MAGIC To achieve the goal,the data was analyzed on Spark platform from perfoming data cleaning,processing to model training with Alternating Least Squares (ALS) algorithm.During which, grid search and cross validation were applied to tune the hyperparameters. It is found that using large rank and iterzations would help achieve a low RMSE. Finally, we choose a rank of 10, iterate 50 times and regPram = 0.15,a RMSE of 0.69 was achieved.This means that on average the model predicts 0.69 above or below values of the original ratings matrix.
# MAGIC 
# MAGIC By successsfully predicting the ratings using the best model, we not only fill the rating dataframe and recommend our users with movies they have never watched, but also find similar movies through their correlations. This brings huge business value to the company.
