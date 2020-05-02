# Prediction-of-House-Value
Background

House is usually the most important and expensive thing being purchased in a person’s life. However, people can only know the estimated price rather than the market price until they purchased it.

Goals

In this project, we used the Zillow housing dataset in Kaggle dataset and we develop different algorithms that can make prediction about the market sale prices with their estimated prices. We will utilize the parallel techniques to reduce the computing time of training process. Finally, the result will be compared in the conclusion.

Step & Result

We carefully select the proper features of the model by investigating the correlation and the significant. By training the regression model, we need to process data first and then choose features.

First, we merge two file by parceid and find there are a lot of NaN values in the dataset. So we drop some columns that NaN values is greater than 70% and add new features by merging some features which are related with others like New_LivingAreaProp, New_zip_count and New_city_count.

The result still have too much feature, and then we get the correlation with the target variable. We choose the features with high correlation values become a new dataset for training and analyze their correlation with each. Obviously, there are some features have high correlation value with others, we trim these features from the training dataset.

In experiments, we investigated the five algorithms and evaluated the loss function of each of them. We utilized the Mean Absolution Error (MAE) as the metric to evaluate the model. Furthermore, the cross validation will also be involved. We used K-fold cross validation and set K equals to 10 for each model. The final score of each Regression is mean value of total MAE.

In this project, we also investigated the effect of using parallel computing. We will use 1, 2, 4, 8, 16 threads in the training process and the result will also be shown.

We use cross_val_score function for each regression and get the MAE score. The Score calculation by the mean value of total sum of |y-y’| which y represent predicted value and y’ stand for real result. In these module, we can see that XGB Regression has highest score we also train each module of different processors.

In the XGB Regression, we use cross validation method to train the model and set the CV argument by 10, but it always cost much time for training model. When the processor is 16, the time will be reduced 86% from only one processor so this is a big advantage when we use multiple processors.

References & Sources                                                                                                         
[1]. Data source : https://www.kaggle.com/c/zillow-prize-1/data                                                             
[2]. Decision tree : https://en.wikipedia.org/wiki/Decision_tree                                                             
[3]. Random Forest : https://towardsdatascience.com/understanding-random-forest-58381e0602d2                                 
[4]. Gradient Boosted Regressor : https://en.wikipedia.org/wiki/Gradient_boosting                                           
[5]. XGBoost Regression: https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost- algorithm-long-she-may-rein-edd9f99be63d                                                                                                                 
[6]. Linear Regression : https://www.statsmodels.org/stable/regression.html                                                 
[7]. neg_mean_squared_error : https://scikitlearn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.m etrics.mean_squared_error  
[8]. Decision tree sample : https://towardsdatascience.com/understanding-random-forest- 58381e0602d2                         
[9]. https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-zillow-prize                                         
[10]. https://github.com/ramborra/Zillow-Home-Value-Prediction/blob/master/Zillow_Code.ipynb?fbclid=IwAR32Iz8fF0Ma--EmvVHwJSKTlIBue35XqpofoMznss0fbckzyDt6x4eTvdQ
