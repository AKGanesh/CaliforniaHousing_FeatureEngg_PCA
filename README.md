

![Logo](https://github.com/AKGanesh/CaliforniaHousing_FeatureEngg_PCA/blob/main/fs.png)


# Feature Engineering on California Housing Dataset
This project tackles predicting California housing prices using machine learning - linear regression in specific. It leverages the scikit-learn library's California housing dataset and explores various feature engineering techniques to optimize model performance. The project implements filter-based methods for feature selection. It also explores dimensionality reduction using Principal Component Analysis (PCA) and incorporates wrapper methods like RFE (Recursive Feature Elimination) and SFE (Sequential Feature Selection)to refine the most influential features for accurate price prediction.

## Implementation Details

- Dataset: [California housing dataset on sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- Model: [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- Input: 8 Features (Median Income, House Age, Average num of Rooms, Average num of Bedrooms, Population, Average num of household members, Latitude, Longitude)
- Output: House Price
- Feature Engineering techniques used: Filter Based (Mutual Info Regression, Pearson Correlation), Wrapper (Recursive Feature Elimination(RFE), Sequential Feature Selection(SFS)) and Dimensionality Reduction (PCA)


## Dataset details
- Number of Instances: 20640

- Number of Attributes: 8 numeric, predictive attributes and the target

- Attribute Information:
    - MedInc        median income in block group
    - HouseAge      median house age in block group
    - AveRooms      average number of rooms per household
    - AveBedrms     average number of bedrooms per household
    - Population    block group population
    - AveOccup      average number of household members
    - Latitude      block group latitude
    - Longitude     block group longitude

- Missing Attribute Values: None

This dataset was obtained from the StatLib repository.
https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

The target variable is the median house value for California districts, expressed in hundreds of thousands of dollars ($100,000).

This dataset was derived from the 1990 U.S. census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

A household is a group of people residing within a home. Since the average number of rooms and bedrooms in this dataset are provided per household, these columns may take surprisingly large values for block groups with few households and many empty houses, such as vacation resorts.

It can be downloaded/loaded using the function
`sklearn.datasets.fetch_california_housing`

- References
Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
Statistics and Probability Letters, 33 (1997) 291-297


## Evaluation and Results


|Method | R2 Score  | MSE  |
|-------| ------------- | ------------- |
|  Filter  MIR - Select Percentile   | 0.58 | 0.56  |
|  Filter  MIR - Select K Best    |0.56       |  0.59|
|  Filter  Pearson - Select K Best    |0.52      |  0.63|
|  Filter  Pandas - Corr    |0.60      |  0.53|
|  Wrapper - RFE   |0.57     |  0.55|
|  Wrapper - SFE   |0.54     |  0.59|
|  Dim. Red - PCA    |0.54      |  0.62|



## Libraries

**Language:** Python

**Packages:** Sklearn, Matplotlib


## Roadmap

- Research on techniques for Logistic Regression

- Need to check the working internals of various techniques



## FAQ

#### Whats is R2 Score?

An R-Squared value shows how well the model predicts the outcome of the dependent variable. R-Squared values range from 0 to 1. An R-Squared value of 0 means that the model explains or predicts 0% of the relationship between the dependent and independent variables.

#### What is MSE?

Model Evaluation is a crucial aspect in the development of a system model. When the purpose of the model is prediction, a reasonable parameter to validate the model’s quality is the mean squared error of prediction.
The value of the error ranges from zero to infinity. MSE increases exponentially with an increase in error. A good model will have an MSE value closer to zero, indicating a better goodness of fit to the data.

#### Why we need to focus on Feature Selection techniques?
The aim of feature selection is to maximize relevance and minimize redundancy.
Feature selection techniques are essential tools for building effective and efficient machine learning models. By strategically selecting the most relevant features, you can enhance model performance, gain better interpretability, improve computational efficiency, and manage data storage requirements.
The way selected features are structured and represented is critical to both quality of the data and the ability to leverage that data for automated decision-making via machine learning.

#### What are Embedded Methods?
Embedded methods offer a convenient and efficient way to perform feature selection while training the model. They are a valuable tool in your machine learning toolbox, but it's important to consider their dependence on the chosen model and potential limitations in interpretability compared to other feature selection techniques. Embedded methods often involve machine learning models with built-in regularization techniques. Examples are LASSO (L1), Ridge (L2) and Decision Trees.

## Acknowledgements

- https://scikit-learn.org/
- https://www.visual-design.net/post/feature-selection-and-eda-in-machine-learning

## Contact

For any queries, please send an email (id on github profile)


##  License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
