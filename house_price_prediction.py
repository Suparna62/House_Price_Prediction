import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data = pd.read_csv("house_prices.csv")

X = data[["area", "bedrooms", "bathrooms", "age", "parking"]]
y = data["price"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


print("Model Accuracy (R²):", round(model.score(X_test, y_test), 3))


example = [[2000, 3, 2, 10, 2]]  
predicted_price = model.predict(example)[0]
print("Predicted Price:", round(predicted_price, 2), "Lakhs")

plt.figure(figsize=(8,5))
plt.scatter(data['area'], data['price'], color='blue', label='Houses')
plt.scatter(example[0][0], predicted_price, color='red', label='Your House', s=100)
plt.title("House Area vs Price")
plt.xlabel("Area (sqft)")
plt.ylabel("Price (Lakhs)")
plt.legend()
plt.grid(True)
plt.show()
