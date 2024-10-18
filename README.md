# Tesla Stock Price Prediction

This project is part of the DATA 170 course at Roanoke College. This will be analyzing the impact of Elon Musk Tweets on Tesla's stock prices. The project processes tweet data, calculates sentiment scores, merges them with stock data, performs feature selection, and builds a linear regression model to predict stock prices.

## Features

- **Data Cleaning and Preprocessing:** Cleans tweet and stock data for analysis.
- **Sentiment Analysis:** Uses TextBlob to calculate sentiment polarity of tweets.
- **Feature Selection:** Utilizes SelectKBest with f_regression to identify significant features.
- **Linear Regression Model:** Predicts Tesla's stock prices based on selected features.
- **Visualization:** Plots actual vs. predicted stock prices for comparison.

## Installation

1. **Clone the repository:**

    ```
    git clone https://github.com/christenseaj1/DATA-170-PROJECT.git
    cd DATA-170-PROJECT
    ```

2. **Create a virtual environment (optional):**

    ```
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**

    ```
    pip install -r requirements.txt
    ```


## Data Setup

This project requires specific datasets for analysis that are **not included in the repository**. You will need to manually download the required data and place them in the `data/` folder.

### Steps to Set Up the Data:

1. **Create a `data/` folder** (if it doesn’t already exist). The folder should be at the root level of the repository, alongside the `src/` folder.

2. **Download the following datasets** from Kaggle:
    - [Elon Musk's Tweets Dataset](https://www.kaggle.com/datasets/marta99/elon-musks-tweets-dataset-2022)
    - [Tesla Stock Data](https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021)

3. **Save the downloaded files**:
    - Rename the files to `cleandata.csv` (for the tweet dataset) and `TSLA.csv` (for the stock data) to match the file names expected by the script.

4. **Place the files** in the `data/` folder. Your folder structure should look like this:
    ```
    DATA-170-PROJECT/
    ├── data/
    │   ├── cleandata.csv      # Cleaned tweet data
    │   ├── TSLA.csv           # Tesla stock data
    ├── src/
    │   └── main.py           
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    └── venv/                 
    ```

## Usage

1. **Prepare Your Data:**
    - Download and place your `cleandata.csv` and `TSLA.csv` files in the `data/` directory.
    - Ensure the CSV files have the necessary columns as expected by the scripts.

2. **Run the Script:**
   - From the `src/` directory, run the script:
     ```bash
     python main.py
     ```

This will process the data, train the model, and display results and a plot comparison of actual vs predicted stock prices.


## Dependencies

- Python 3.x
- numpy
- pandas
- textblob
- matplotlib
- scikit-learn

## Project Structure
```
DATA-170-PROJECT/
│
├── data/                     
│   ├── cleandata.csv
│   ├── TSLA.csv
│
├── src/                      
│   └── main.py               
│
├── .gitignore               
├── README.md                 
├── requirements.txt         
└── venv/                     
```


