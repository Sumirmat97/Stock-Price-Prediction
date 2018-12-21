# Stock Price Prediction

## About Project ##
It is a stock price prediction application made in Python.

The users can either train the models with their data and then use the for predictions or predict the stock prices of "Bank of Baroda" using the existing models.
### To train models and predict prices: ###
* Provide the past data of that stock you want to predict in csv format in trainStockData.csv and some data for testing the models in testStockData.csv, the columns should be: "date","open","low","high","close","adj_close","volume". 
(We have used 3 years data in traing and 1 year data for testing of the models)
* Provide the corresponding economical news for that stock in csv format in trainNewsData.csv and some data for testing the models in testNewsData.csv, the columns should be: "date","news","close". "close" is the closing price of that stock on the specified date.
#### Note: #### The dates in news train and test files should be same and in same order.
* The command to run the application is: python main.py --stockfilepath ..\trainStockData.csv --teststockfilepath ..\testStockData.csv --newsfilepath ..\trainNewsData.csv --testnewsfilepath ..\testNewsData.csv
* The application will train models and also show the grpahs after testing the models on test data.

### To predict for cetarin days using the models: ###
* 

Emoticons require net connection (as we have integrated using Emojione project)

Messages can be starred and starred messages can be seen separately.
Individual messages can be deleted. For both of these an interactive and easy to use GUI is provided.

Users can chat in incognito mode. The mode deletes the messages sent by the user after the user signs out.

## How do I set up ##
* Required postgreSql and an Apache server.
* Create database by importing the .backup file present in the repository.
* Change the credentials in src/getConnection.php
* Run the apache server and run index.php file in web folder.

## Who do I talk to ? ##
* Repo owner
* email: sumir.mathur8@gmail.com
