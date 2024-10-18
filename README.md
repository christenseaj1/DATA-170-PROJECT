# Tesla Stock Price Prediction

This project analyzes Twitter sentiment and its impact on Tesla's stock prices. It processes tweet data, calculates sentiment scores, merges them with stock data, performs feature selection, and builds a linear regression model to predict stock prices.

## Features

- **Data Cleaning and Preprocessing:** Cleans tweet and stock data for analysis.
- **Sentiment Analysis:** Uses TextBlob to calculate sentiment polarity of tweets.
- **Feature Selection:** Utilizes SelectKBest with f_regression to identify significant features.
- **Linear Regression Model:** Predicts Tesla's stock prices based on selected features.
- **Visualization:** Plots actual vs. predicted stock prices for comparison.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/tesla-stock-prediction.git
    cd tesla-stock-prediction
    ```

2. **Create a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare Your Data:**
    - Download the CSV files from kaggle.com 
    ```bash
   https://www.kaggle.com/datasets/marta99/elon-musks-tweets-dataset-2022
   https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021
    ```
    - Place your `cleandata.csv` and `TSLA.csv` files in the `data/` directory.
    - Ensure the CSV files have the necessary columns as expected by the scripts.

2. **Run the Main Script:**

    ```bash
    python src/main.py
    ```

    This will process the data, train the model, and display a plot comparing actual and predicted stock prices.

## Dependencies

- Python 3.x
- numpy
- pandas
- textblob
- matplotlib
- scikit-learn

## Project Structure
```plaintext
tesla-stock-prediction/
│
├── data/                     # Folder for storing data files (e.g., cleandata.csv, TSLA.csv)
│   ├── cleandata.csv
│   ├── TSLA.csv
│
├── src/                      # Source code directory containing the main script
│   └── main.py               # Main Python script for processing data and making predictions
│
├── .gitignore                # Specifies files and directories to be ignored by Git
├── README.md                 # Project documentation
├── requirements.txt          # Lists Python dependencies for the project
└── venv/                     # Virtual environment (optional, if created)

