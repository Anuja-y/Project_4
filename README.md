# Australian Mining Comapany Stock Market Analysis

Objective

This project aims to analyze and predict the stock performance of mining companies in Australia. The stock market, with its allure of high returns, also comes with its fair share of risks. Consequently, rigorous analysis is imperative for informed investment decisions. Considering the pivotal role that the mining sector holds in the Australian economy, our study seeks to shed light on market dynamics and evolving trends. While our primary focus is on the BHP Group, the code is designed to be adaptable, allowing users to switch to their company of interest.

Machine Learning Workflow

- Data Collection & Exploration 
- Data Visualization
- Feature selection
- Split the data into Train, Test and Validation sets
- Implementing model prediction and evaluation

Data

To obtain the essential market data for our stock prediction model, we will employ the yFinance library in Python. This library is made for fetching pertinent data for any given ticker symbol from the Yahoo Finance website. The yFinance library allows us to seamlessly acquire the most recent market data and integrate it into our model.

Data Visualization
In the project, we used two visualization tools: Tableau for its user-friendly interface and the library Matplotlib in Python.


Features Analysed
- Date :  Calendar date of the trading day. 
- Open : Opening price of the trading day. 
- High : Highest price of the stock traded during the day.
- Low : Lowest price of the stock traded during the day.
- Close : Closing price of the trading day.
- Adj Close : Adjusted closing price of the trading day.
- Volume : Number of shares traded in exchange during the day.


Methodology

Our analysis uses a combination of statistical modeling and machine learning. By using methods such as ARIMA for time series forecasting and Random Forest Regression for prediction, we aim to capture the nuances and intricacies of stock movements.

Technical indicators
RSI (Relative Strength Index): Measures the momentum of price movements to identify overbought or oversold conditions.
SMA (Simple Moving Average): Averages stock prices over a specific period to identify trends by smoothing out price fluctuations.
Standard Deviation: Assesses price volatility by determining the variation of stock prices from their average

Implementation
Comparing benchmark and proposed models.

Procedure:
- Retrieve the raw data.
- Engage in feature selection, emphasizing technical indicators.
- Determine the relevant features and target dataframe.
- Normalize the dataset.
- Partition the data into training, testing, and validation subsets.
- Assess the performance of the models.
- Summarize our findings in the conclusion.

Optimal Model Selection
- After evaluating both the RMSE Score and R^2 Score, the SVM model with fine-tuning emerges as the superior choice for the project

Limitations of stock Prediction Model  

- The most foundational limitations is that past performance does not guarantee future results . Even if our model captures historical trends perfectly, unpredictable events can always affect stock prices. Sudden shifts in the economy, like recessions or booms can change the stock game .


Conclusion

- While predictive models offer valuable insights into stock market trends, they should be viewed as one component of an investor's decision-making process. It's essential to integrate these model insights with expert guidance, thorough research, and an understanding of the broader market context.

