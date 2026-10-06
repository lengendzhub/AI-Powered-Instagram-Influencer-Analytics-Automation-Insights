# ==============================================================================
# SIMPLE MACHINE LEARNING TUTORIAL
# 
# This is a self-contained script to show you how easy it is to build a Machine
# Learning model in Python. We will build a model to predict house prices 
# based on their size (square feet) and number of bedrooms.
#
# Run this by typing: python simple_ml_tutorial.py
# ==============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

print("STEP 1: GET THE DATA")
print("--------------------")
# In real life, you load this from a CSV. Here, we'll create a tiny fake dataset.
# The data has 'Size', 'Bedrooms', and the actual 'Price' of the house.
data = {
    'Size_sqft': [1000, 1500, 2000, 1200, 2500, 1800, 2200, 1100],
    'Bedrooms':  [2,    3,    4,    2,    4,    3,    4,    2],
    'Price_USD': [150000, 220000, 310000, 170000, 350000, 280000, 320000, 165000]
}
df = pd.DataFrame(data)
print("Our Dataset:")
print(df)
print("\n")


print("STEP 2: SPLIT INTO 'FEATURES' (X) AND 'TARGET' (y)")
print("--------------------------------------------------")
# X (Features) = The clues we give the model
X = df[['Size_sqft', 'Bedrooms']]

# y (Target) = The answer we want the model to learn to predict
y = df['Price_USD']
print("Features (X):")
print(X.head(2))
print("Target (y):")
print(y.head(2))
print("\n")


print("STEP 3: HIDE SOME DATA FOR TESTING")
print("----------------------------------")
# We split our data: 80% to train (teach) the model, 20% to test it later.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"We have {len(X_train)} houses to train on, and {len(X_test)} houses to test on.\n")


print("STEP 4: CREATE AND TRAIN THE MODEL")
print("----------------------------------")
# Create a blank model
model = LinearRegression()

# TRAIN IT! This is where the machine "learns" the math relationship
model.fit(X_train, y_train)
print("Model trained successfully! (It found the hidden patterns)\n")


print("STEP 5: TEST THE MODEL WITH UNSEEN DATA")
print("---------------------------------------")
# We give the model the X_test data (which it hasn't seen yet) and ask it to guess the prices
predictions = model.predict(X_test)

# Let's compare the model's guesses to the actual real prices
results = pd.DataFrame({
    'Real_Price': y_test, 
    'Models_Guess': predictions
})
# Format the numbers nicely
results['Real_Price'] = results['Real_Price'].map('${:,.0f}'.format)
results['Models_Guess'] = results['Models_Guess'].map('${:,.0f}'.format)
print("Test Results:")
print(results)
print("\n")


print("STEP 6: USE THE MODEL ON BRAND NEW DATA")
print("---------------------------------------")
# Imagine a customer comes in with a brand new house to sell: 
# It is 1,600 sq ft and has 3 bedrooms. What should the price be?
new_house = pd.DataFrame({'Size_sqft': [1600], 'Bedrooms': [3]})

predicted_price = model.predict(new_house)[0]
print(f"For a 1,600 sq ft, 3-bedroom house...")
print(f"The AI predicts the price should be: ${predicted_price:,.2f}")
