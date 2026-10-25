
import numpy as np

print("NumPy version:", np.__version__)

# STEP 2: Create the original dataset
house_sizes = np.array([
    500, 800, 1000, 1200, 1500,
    1800, 2000, 2200, 2500, 2800
])

house_prices = np.array([
    15, 24, 30, 36, 45,
    54, 60, 66, 75, 84
])

print("\n--- STEP 2: DATASET ---")
print("House sizes:", house_sizes)
print("House prices:", house_prices)
print("Number of houses:", house_sizes.size)

print("Size array shape:", house_sizes.shape)
print("Price array shape:", house_prices.shape)

print("\nThird house:")
print("Size:", house_sizes[2], "sq ft")
print("Price:", house_prices[2], "lakhs")

# Demonstrate adding an eleventh house.
# Keep the original ten-house arrays unchanged for later ML steps.
expanded_sizes = np.append(house_sizes, 3000)
expanded_prices = np.append(house_prices, 90)

print("\nDataset after adding one house:")
print("Updated number of houses:", expanded_sizes.size)
print("Updated sizes:", expanded_sizes)
print("Updated prices:", expanded_prices)

# STEP 3: Explore the data
def explore_data(name, data):
    print(f"\n--- {name} ---")
    print("Mean:", np.mean(data))
    print("Minimum:", np.min(data))
    print("Maximum:", np.max(data))
    print("Median:", np.median(data))
    print("Standard deviation:", np.std(data))


print("\n--- STEP 3: EXPLORATORY DATA ANALYSIS ---")

explore_data("House sizes (square feet)", house_sizes)
explore_data("House prices (lakhs)", house_prices)


# STEP 4: Investigate the relationship

print("\n--- STEP 4: SIZE AND PRICE RELATIONSHIP ---")

# Choose two houses from the original dataset
size_a = house_sizes[1]    # 800 sq ft
price_a = house_prices[1]  # 24 lakhs

size_b = house_sizes[7]    # 2200 sq ft
price_b = house_prices[7]  # 66 lakhs

# Calculate the differences
size_difference = size_b - size_a
price_difference = price_b - price_a

print("First house:", size_a, "sq ft,", price_a, "lakhs")
print("Second house:", size_b, "sq ft,", price_b, "lakhs")

print("Difference in size:", size_difference, "sq ft")
print("Difference in price:", price_difference, "lakhs")

# Calculate slope (price change per square foot)
slope = price_difference / size_difference

# Calculate intercept
intercept = price_a - slope * size_a

print("Slope:", slope, "lakhs per sq ft")
print("Intercept:", intercept)

# Predict the price of a 1300 sq ft house
new_size = 1300
predicted_price = slope * new_size + intercept

print("Predicted price for", new_size, "sq ft:",
      predicted_price, "lakhs")

# STEP 5: Split data into training and testing sets

print("\n--- STEP 5: TRAINING AND TESTING DATA ---")

# First 8 houses are used for training
X_train = house_sizes[:8]
y_train = house_prices[:8]

# Last 2 houses are used for testing
X_test = house_sizes[8:]
y_test = house_prices[8:]

# Print all four arrays
print("X_train (training sizes):", X_train)
print("y_train (training prices):", y_train)

print("X_test (testing sizes):", X_test)
print("y_test (testing prices):", y_test)

# Count the examples
print("\nNumber of training examples:", len(X_train))
print("Number of testing examples:", len(X_test))

# Check that sizes match their corresponding prices
print("\nTraining data pairs:")
for size, price in zip(X_train, y_train):
    print(size, "sq ft ->", price, "lakhs")

print("\nTesting data pairs:")
for size, price in zip(X_test, y_test):
    print(size, "sq ft ->", price, "lakhs")
# STEP 6: Linear Regression using NumPy

print("\n--- STEP 6: TRAINING MODEL ---")

x_mean = np.mean(X_train)
y_mean = np.mean(y_train)

numerator = np.sum(
    (X_train - x_mean) * (y_train - y_mean)
)

denominator = np.sum(
    (X_train - x_mean) ** 2
)

w = numerator / denominator
b = y_mean - w * x_mean

print("Mean of training sizes:", x_mean)
print("Mean of training prices:", y_mean)

print("Slope (w):", w)
print("Intercept (b):", b)


def predict_price(size):
    return w * size + b


print(
    "Predicted price for 1300 sq ft:",
    predict_price(1300),
    "lakhs"
)
# STEP 7: Evaluate the model

print("\n--- STEP 7: MODEL EVALUATION ---")

y_pred = predict_price(X_test)

errors = y_test - y_pred

mae = np.mean(np.abs(errors))
mse = np.mean(errors ** 2)

print("Actual prices:", y_test)
print("Predicted prices:", y_pred)
print("Prediction errors:", errors)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
# STEP 8: Interactive prediction

print("\n--- STEP 8: PREDICT HOUSE PRICE ---")

try:
    user_size = float(
        input("Enter house size in square feet: ")
    )

    if user_size <= 0:
        print("Please enter a positive house size.")
    else:
        result = predict_price(user_size)

        print(
            f"Predicted house price: ₹{result:.2f} lakh"
        )

except ValueError:
    print("Invalid input! Please enter a number.")
# CHALLENGE A: Unusual house

print("\n--- CHALLENGE A: OUTLIER ---")

unusual_prices = np.append(house_prices, 200)

print("Original mean:", np.mean(house_prices))
print("New mean:", np.mean(unusual_prices))

print("Original median:", np.median(house_prices))
print("New median:", np.median(unusual_prices))
# CHALLENGE C: Compare with baseline

print("\n--- CHALLENGE C: BASELINE COMPARISON ---")

baseline_predictions = np.full(
    len(y_test),
    np.mean(y_train)
)

baseline_mae = np.mean(
    np.abs(y_test - baseline_predictions)
)

model_mae = np.mean(
    np.abs(y_test - y_pred)
)

print("Baseline predictions:", baseline_predictions)
print("Baseline MAE:", baseline_mae)
print("Linear regression MAE:", model_mae)

if model_mae < baseline_mae:
    print("Linear regression performs better.")
elif model_mae == baseline_mae:
    print("Both models have equal MAE.")
else:
    print("Baseline performs better.")
