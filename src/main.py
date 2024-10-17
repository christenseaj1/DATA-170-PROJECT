import numpy as np
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates


#--- Clean Tweet Data ---
df_tweets = pd.read_csv('cleandata.csv')

#Remove all other columns except Date and Cleaned_Tweets
df_tweets = df_tweets[['Date', 'Cleaned_Tweets']]

#Rearrange Date format and remove time
df_tweets['Date'] = pd.to_datetime(df_tweets['Date']).dt.date
#------------------------------------------------------------



#--- Sentiment Scoring ---
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

#Adds the new sentiment data into a new column "Sentiment"
df_tweets['Sentiment'] = df_tweets['Cleaned_Tweets'].apply(get_sentiment)
df_tweets.to_csv('cleandata.csv', index = False)

#Average Sentiment Score for each Date
avg_sentiments = df_tweets.groupby('Date', as_index=False).agg({'Sentiment':'mean'})
#-----------------------------------------------------------------------------------
#print(df_tweets)
#print(avg_sentiments)



#--- Clean Tesla Stock Data ---
df_stocks = pd.read_csv('TSLA.csv')

#Rearrange Date format
df_stocks['Date'] = pd.to_datetime(df_stocks['Date']).dt.date

#Remove dates and keep only ones in "avg_sentiment"
df_stocks = df_stocks[df_stocks['Date'].isin(avg_sentiments['Date'])]
df_stocks.to_csv('TSLA.csv', index=False)
#--------------------------------------------------------------------


#--- Feature Engineering by adding the Sentiment Score to the Stock Data ---
df_combined = pd.merge(df_stocks, avg_sentiments, on='Date', how='inner')
if 'Sentiment' not in df_stocks.columns:
    df_combined.to_csv('TSLA.csv', index=False)
#---------------------------------------------------------------------------
#print(df_stocks)


#--- Feature Select Tesla Stock Data ---

X = df_stocks.drop(['Date', 'Close', 'Adj Close'], axis=1)
y = df_stocks['Close']

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X.assign(Date=df_stocks['Date']), y, test_size=.2, random_state=42)
X_test, test_dates = X_test.drop('Date', axis=1), X_test['Date']

#SelectKBest with f_regression as the score function
selector = SelectKBest(score_func=f_regression, k='all')
X_train, train_dates = X_train.drop('Date', axis=1), X_train['Date']

# Fit and transform the data
X_train_transformed = selector.fit_transform(X_train, y_train)
X_test_transformed = selector.transform(X_test)
selected_features = X.columns[selector.get_support(indices=True)]

#Fit the SelectKBest method to the training data
selector.fit(X_train, y_train)

#Get the scores for each feature
scores = selector.scores_
feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': scores})
print(feature_scores.sort_values(by='Score', ascending=False))
#---------------------------------------------------------------------


#--- Graph the Data ---
# Create a Linear Regression model
model = LinearRegression()
model.fit(X_train_transformed, y_train)
y_pred = model.predict(X_test_transformed)

#Create a DataFrame for the actual and predicted stock prices
df_results = pd.DataFrame({'Date': test_dates, 'Actual': y_test, 'Predicted': y_pred})

#Plot the actual stock prices
df_results.sort_values('Date', inplace=True)
plt.figure(figsize=(10,5))
plt.plot(df_results['Date'], df_results['Actual'], label='Actual Stock Price')
plt.plot(df_results['Date'], df_results['Predicted'], label='Predicted Stock Price')

#Set the title and labels
plt.title('Tesla Stock Prices: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Price')

#Format the Graph
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator())
plt.gcf().autofmt_xdate()
plt.legend()
plt.show()