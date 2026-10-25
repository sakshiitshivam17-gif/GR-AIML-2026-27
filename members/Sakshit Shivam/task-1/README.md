# House Price Prediction using NumPy

## About the Project
This is my Task 1 mini-project for the AI-ML department. I used NumPy to work with house sizes and their prices and made a simple program to predict the price of a house based on its size.

## What I Did
- Created NumPy arrays for house sizes and prices.
- Found the mean, median, minimum, maximum and standard deviation.
- Used the first 8 houses for training and the last 2 for testing.
- Made a simple linear regression model using NumPy.
- Checked the model's predictions using MAE and MSE.
- Added an input option so the user can enter a house size and get its predicted price.

## Results
- The slope came out to 0.03.
- A house of 1300 sq ft is predicted to cost ₹39 lakh.
- The model's MAE and MSE on the test data were both 0.
- I also tried adding an outlier and compared the model with a simple baseline.

## What I Learned
While doing this project, I got practice with NumPy arrays, basic data analysis and how a simple prediction model works. I also learned that outliers can affect the data and that a model performing well on a small dataset does not mean it will always work well on real house prices.

## How to Run
Install NumPy first:

```bash
python -m pip install numpy
```

Then run:

```bash
python house_price_prediction.py
```
