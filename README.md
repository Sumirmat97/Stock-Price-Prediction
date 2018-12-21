# Stock Price Predictor
A stock price redictor that combines the results of prediction from time series analysis and news mining.

## About Project ##
It is a stock price prediction application made in Python.

The users can either train the models with their data and then use the for predictions or predict the stock prices of "Bank of Baroda" using the existing models.
### To train models and predict prices: ###
1. Provide the past data of that stock you want to predict in csv format in trainStockData.csv and some data for testing the models in testStockData.csv, the columns should be: "date","open","low","high","close","adj_close","volume". 
(We have used 3 years data in traing and 1 year data for testing of the models)
1. Provide the corresponding economical news for that stock in csv format in trainNewsData.csv and some data for testing the models in testNewsData.csv, the columns should be: "date","news","close". "close" is the closing price of that stock on the specified date. 
(We suggest 10 news for each day should be appended and then given in same column.)
  **Note:  The dates in news train and test files should be same and in same order as in train and test files of the stock's data.**
1. The command to run the application is (in src folder): 
```
python main.py --stockfilepath ..\trainStockData.csv --teststockfilepath ..\testStockData.csv --newsfilepath ..\trainNewsData.csv --testnewsfilepath ..\testNewsData.csv
```
The application will train models and also show the graphs after testing the models on test data.

### To predict for cetarin days using the models: ###
1. Provide the data of the stock for which the models were trained in predictStockData.csv, the columns should be: "date","open","low","high","close","adj_close","volume".
1. Provide the corresponding economical news for that stock in predictNewsData.csv, the columns should be: "date","news". Make sure the dates are same and in same order.
1. The application chooses the models which have the least mean absolute error for prediction. 
1. The command to run the application is for prediction is (in src folder):
```
python main.py --predict 1 --stockfilepath ..\predictStockData.csv --newsfilepath ..\predictNewsData.csv
```
**Note: The users may also specify the model or models they want to use by giving them in command line arguments.**
```
python main.py --predict 1 --stockfilepath ..\predictStockData.csv --newsfilepath ..\predictNewsData.csv --modeltimeseries ..\Models\<model_name> --modelnews ..\Models\<model_name>
```
Either of the models are optional.

## How do I set up ##
1. Clone or download the project
1. Python 3.6 or higher is required 
1. Run the command: 
``` pip install -r requirements.txt ```

## Who do I talk to ? ##
* Repo owners
* email: sumir.mathur8@gmail.com
