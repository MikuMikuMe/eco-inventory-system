Creating an eco-inventory system that uses machine learning to optimize stock levels involves several components: data handling, building the machine learning model, and implementing the inventory management logic. Below is a simplified version of such a system. This example will focus on using historical sales data to predict future demand, thereby optimizing stock levels. 

```python
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import logging

# Set up basic configurations for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Sample data generation function
def generate_sample_data(n):
    """Generates sample inventory data for demonstration purposes."""
    dates = pd.date_range(start='2022-01-01', periods=n, freq='D')
    demand = np.random.poisson(20, n)  # Generate random demand data
    lead_time = np.random.randint(1, 10, n)  # Random lead times between 1 to 10 days
    inventory_data = pd.DataFrame({
        'Date': dates,
        'Demand': demand,
        'LeadTime': lead_time
    })
    return inventory_data

# Data preparation
def prepare_data(inventory_data):
    """Prepares training and testing data from the inventory data."""
    inventory_data['DayOfWeek'] = inventory_data['Date'].dt.dayofweek
    X = inventory_data[['DayOfWeek', 'LeadTime']]
    y = inventory_data['Demand']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
def train_model(X_train, y_train):
    """Trains a machine learning model on the inventory data."""
    model = LinearRegression()
    try:
        model.fit(X_train, y_train)
        logging.info("Model training successful.")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return None

# Demand prediction
def predict_demand(model, X):
    """Uses the trained model to predict demand."""
    try:
        return model.predict(X)
    except Exception as e:
        logging.error(f"Error during making predictions: {e}")
        return None

# Inventory optimization
def optimize_inventory(predictions, current_stock):
    """Calculates optimized stock levels based on demand predictions."""
    try:
        # Assuming a simple restock strategy based on predicted demand
        reorder_point = predictions.mean() + 1.96 * predictions.std() / np.sqrt(len(predictions))
        optimized_stock = reorder_point - current_stock
        optimized_stock = np.where(optimized_stock < 0, 0, optimized_stock)
        logging.info(f"Optimized stock level calculated: {optimized_stock}")
        return optimized_stock
    except Exception as e:
        logging.error(f"Error during inventory optimization: {e}")
        return None

# Main function
def main():
    # Generate sample inventory data
    inventory_data = generate_sample_data(100)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(inventory_data)

    # Train model
    model = train_model(X_train, y_train)

    # Check if model is trained
    if model is None:
        logging.error("Model training failed, terminating program.")
        return

    # Predict future demand
    predictions = predict_demand(model, X_test)

    # Check if predictions are made
    if predictions is None:
        logging.error("Prediction failed, terminating program.")
        return

    # Simply assuming current stock for simplicity
    current_stock = 50

    # Optimize inventory
    optimized_stock = optimize_inventory(predictions, current_stock)

    # Display results
    if optimized_stock is not None:
        logging.info(f"Recommended stock level adjustment: {optimized_stock}")

if __name__ == '__main__':
    main()
```

### Explanation:
- **Logging:** Used to track the flow and errors during execution.
- **Data Generation:** A function to generate random inventory data for demonstration.
- **Model Training:** Uses a linear regression model to predict demand based on the day of the week and lead time.
- **Prediction and Optimization:** Predicts future demand and calculates the optimized stock level.
- **Error Handling:** Uses try-except blocks to catch and log errors during model training, predictions, and optimization phases.

This is a basic framework for an eco-inventory system and uses random data for illustration. For a real system, you would need a proper dataset, possibly more sophisticated models, and a connection to a database or API for dynamic data handling.