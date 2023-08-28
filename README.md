# Australian Mining Company Stock Market Analysis

Objective

This project aims to analyze and predict the stock performance of mining companies in Australia using various machine learning methods to ensure optimal results. The Australian stock market, known for its potential for high returns and inherent risks, necessitates a rigorous analysis to make informed investment decisions. Considering the pivotal role the mining sector plays in the Australian economy, our research endeavors to provide critical insights into market dynamics and emerging trends.

Machine Learning Workflow

- Data Collection & Exploration
- Feature selection
- Data Visualization
- Split the data into Train, Test and Validation sets
- Implementing model prediction and evaluation

Data

To obtain the essential market data for our stock prediction model, we will employ the yFinance library in Python. This library is made for fetching pertinent data for any given ticker symbol from the Yahoo Finance website. The yFinance library allows us to seamlessly acquire the most recent market data and integrate it into our model.

Data Visualization
In the project, we used two visualization tools: Tableau for its user-friendly interface and the library Matplotlib in Python.
https://prod-apsoutheast-a.online.tableau.com/#/site/flormartinauss/workbooks/329434?:origin=card_share_link 


Features Analysed
First Part 
- Date :  Calendar date of the trading day. 
- Open : Opening price of the trading day. 
- High : Highest price of the stock traded during the day.
- Low : Lowest price of the stock traded during the day.
- Close : Closing price of the trading day.
- Adj Close : Adjusted closing price of the trading day.
- Volume : Number of shares traded in exchange during the day.

  Second Part
  In addition to the features we have used in the first part we have added these technical indicators to improve the model performance:
- RSI measures momentum
- SMA calculates trend, 
- Standard Deviation quantifies volatility.


Methodology

Our analysis uses a combination of statistical modeling and machine learning. By employing methods LSTM for time series forecasting and Random Forest Regression for prediction, we aim to capture the nuances and complexities of stock movements. Additionally, we utilize technical indicators such as the Relative Strength Index, Simple Moving Average, and Standard Deviation.


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
- Based on our label data and the features considered, the SVR emerged as the best model, especially when evaluating the RMSE and R2 scores.

Limitations of stock Prediction Model  

- The most foundational limitations is that past performance does not guarantee future results . Even if our model captures historical trends perfectly, unpredictable events can always affect stock prices. Sudden shifts in the economy, like recessions or booms can change the stock game .


Conclusion
- Based on our label data and the features considered, the SVR emerged as the best model, especially when evaluating the RMSE and R2 scores.
While predictive models offer valuable insights into stock market trends, they should be viewed as one component of an investor's decision-making process. It's essential to integrate these model insights with expert guidance, thorough research, and an understanding of the broader market context.


