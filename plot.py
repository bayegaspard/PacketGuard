import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for headless environments
import matplotlib.pyplot as plt


# Load the data from CSV
file_path = 'results/graph_data.csv'  # Replace with the actual file path
data = pd.read_csv(file_path)

# Filter for training loss and validation metrics
training_data = data[(data['Metric_Type'] == 'Training') & (data['Metric_Name'] == 'Loss')]
validation_data = data[data['Metric_Type'] == 'Validation']

# Plot Training Loss over Epochs
plt.figure(figsize=(10, 6))
plt.plot(training_data['Epoch'], training_data['Metric_Value'], label='Training Loss', marker='o')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('results/training_loss_plot.png')
plt.show()

# Plot Validation Metrics over Epochs
plt.figure(figsize=(10, 6))
for metric in validation_data['Metric_Name'].unique():
    subset = validation_data[validation_data['Metric_Name'] == metric]
    plt.plot(subset['Epoch'], subset['Metric_Value'], label=metric, marker='o')

plt.title('Validation Metrics over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.savefig('results/validation_metrics_plot.png')
plt.show()

import seaborn as sns

# Load the CSV data

# Filter for sensitivity data
sensitivity_data = data[data["Metric_Type"] == "Sensitivity"]

# Sort by sensitivity value and limit the number of features displayed
sensitivity_data = sensitivity_data.sort_values("Metric_Value", ascending=False).head(30)

# Plot sensitivity analysis
plt.figure(figsize=(12, 8))
sns.barplot(x='Metric_Value', y='Metric_Name', data=sensitivity_data)
plt.xlabel("Sensitivity Value", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Top Features by Sensitivity Analysis", fontsize=16)
plt.tight_layout()

# Save the plot
plt.savefig("results/sensitivity_analysis.png", dpi=300)

print("Sensitivity analysis plot saved as 'sensitivity_analysis.png'")