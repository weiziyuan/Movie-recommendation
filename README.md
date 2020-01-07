# Movie Recommendation Engine Building in Apache Spark
code link:

https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2268229575846883/1611422526940121/6723471235902913/latest.html
## Overview
In this project, the movie data is movie lens data set which includes about 600 users and 9500 movies.

The purpose of the project are to:

* find similar movies.

* gain insights on movie recommendations to users.


To achieve the goal, we will:

* Train the model with Alternating Least Squares (ALS) algorithm.

* Predict movie rating and recommend movies to users.

* Find the correlations between different movies and infer similarities.

This will be implemented on Spark due to its fast speed for large-scale data processing and readiness to use.


## Take a glance at the data
![alt text](https://github.com/weiziyuan/Movie-recommendation/blob/master/image/movie_table.png)
![alt text](https://github.com/weiziyuan/Movie-recommendation/blob/master/image/rating_table.png)
The rating dataframe is 98.3% empty.

Histogram of the user rating count
![alt text](https://github.com/weiziyuan/Movie-recommendation/blob/master/image/hist_rating_cnt.png)

Histogram of the average user rating

![alt text](https://github.com/weiziyuan/Movie-recommendation/blob/master/image/hist_avg_rating.png)

Most movie ratings lie in 1.5-4.5.

## Model performance
RMSE = 0.687 on all data, saying that on average the model predicts 0.69 above or below values of the original ratings matrix.

* Predicitng movie(top) and user(bottom) rating

![alt text](https://github.com/weiziyuan/Movie-recommendation/blob/master/image/mp_alldata_movie.png)
![alt text](https://github.com/weiziyuan/Movie-recommendation/blob/master/image/mp_alldata_user.png)

## Movie recommendation
We first recommend movies to user 414 and 599 who rated largest number of movies(2698 and 2478).
![alt text](https://github.com/weiziyuan/Movie-recommendation/blob/master/image/user414.png)
![alt text](https://github.com/weiziyuan/Movie-recommendation/blob/master/image/user599.png)

## Finding similar movies
We find similar movie based on the correlation between movie ratings, the closer the correlation between movie A and B to 1, the more similar they are.

However, some movies have very few ratings and may end with high correlation simply because only one or two users gave them similar ratings. These are not reliable results.

We saw a sharp decline in number of ratings from 100.Therefore, we will choose this as our threshold. This is saying, we will only find similar movies that have been rated more than 100 times


![alt text](https://github.com/weiziyuan/Movie-recommendation/blob/master/image/count_vs_rating.png)

We pick movie 471 and see other movies' rating correlation with it.
![alt text](https://github.com/weiziyuan/Movie-recommendation/blob/master/image/movie471_corr.png)

Simlar movies to movie 471

![alt text](https://github.com/weiziyuan/Movie-recommendation/blob/master/image/similar_to_471.png)

## Summary
Overall, the model did a good job on predicting the user rating, specifically, between 1.5-4.5. However, when predicting extreme scores(either low or high) like 0.5 or 5, it needs improvement. 

Two possible reasons:
* most ratings are between 2-4(we can tell from the avg(rating)-count plot), which means extreme high or low ratings are sparse, therefore, to accurately predict ratings in this region may be more difficult.
* We know that the dataframe is highly sparse(98.3%), again bring challenges to the prediction. 

Suggestions :
* We could further increase iteration to lower the RMSE, however, due to long running time, we didn’t try it here. Also, we noticed before, this didn’t really help too much.
* Collect more data.
For now, it is not a good idea to use a more complex model due to limited ratings. 
In the future, we could encourage users to rate the film for low sparsity. With larger dataset, we could use a more complex model, for example, neural network for better performance.
