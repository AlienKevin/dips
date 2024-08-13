import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import stats

# Read the CSV file
data = pd.read_csv('validation_points.tsv', sep='\t')

# Remove the first 50% of the data points
num_points_to_remove = int(len(data) * 0.5)
data = data.iloc[num_points_to_remove:]

# Reset the index after removing points
data = data.reset_index(drop=True)

print(f"Removed {num_points_to_remove} data points (10% of the original data)")
print(f"New data shape: {data.shape}")


# Define a list of functions to try
def inverse_exp(x, a, b, c):
    return a * np.exp(-b / x) + c

def power_law(x, a, b, c):
    return a * x**b + c

def logarithmic(x, a, b, c):
    return a * np.log(x) + b

functions = [inverse_exp, power_law, logarithmic]
function_names = ['Inverse Exponential', 'Power Law', 'Logarithmic']

# Find the best fitting function
best_r_squared = -np.inf
best_function = None
best_popt = None
best_function_name = None

for func, name in zip(functions, function_names):
    try:
        popt, _ = curve_fit(func, data['step'], data['validation_loss'])
        y_pred = func(data['step'], *popt)
        r_squared = stats.pearsonr(data['validation_loss'], y_pred)[0]**2
        
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_function = func
            best_popt = popt
            best_function_name = name
    except:
        continue

print(f"Best fitting function: {best_function_name}")
print(f"R-squared: {best_r_squared:.4f}")

target_steps = 104000 * 8

# Generate points for plotting
x_fit = np.linspace(data['step'].min(), target_steps, 1000)
y_fit = best_function(x_fit, *best_popt)

# Plot the data and the fitted curve
plt.figure(figsize=(10, 6))
plt.scatter(data['step'], data['validation_loss'], label='Data')
plt.plot(x_fit, y_fit, 'r-', label='Fitted Curve')
plt.xlabel('Step')
plt.ylabel('Validation Loss')
plt.title(f'{best_function_name} Fit of Validation Loss')
plt.legend()

# Predict the validation loss at 1,040,000 steps
predicted_loss = best_function(target_steps, *best_popt)

print(f"Predicted validation loss at 1,040,000 steps: {predicted_loss:.6f}")

# Add the prediction point to the plot
plt.scatter(target_steps, predicted_loss, color='green', s=100, label='Prediction')
plt.legend()

plt.show()
