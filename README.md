# Movie Recommendation Engine Building in Apache Spark
## Overview
In this project, the movie data is movie lens data set which includes about 600 users and 9500 movies.

The purpose of the project are to:

* find similar movies.

* gain insights on movie recommendations to users.


To achieve the goal, we will:

* Train the regression model with Alternating Least Squares (ALS) algorithm.

* Predict movie rating and recommend movies to users.

* Find the correlations between different movies and infer similarities.

This will be implemented on Spark due to its fast speed for large-scale data processing and readiness to use.


## Take a glance at the data
![alt text](https://github.com/weiziyuan/Movie-recommendation/blob/master/image/movie_table.png)
![alt text](https://github.com/weiziyuan/Movie-recommendation/blob/master/image/rating_table.png)
The rating dataframe is 98.3% empty.

## Model performance
RMSE = 0.687 on all data, saying that on average the model predicts 0.69 above or below values of the original ratings matrix.

![alt text](https://github.com/weiziyuan/Movie-recommendation/blob/master/image/mp_alldata_movie.png)

## Movie recommendation
We first recommend movies to user 414 who rated largest number of movies(2698).


