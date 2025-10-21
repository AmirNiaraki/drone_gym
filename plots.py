import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

# Get the current directory
directory = "/Users/jherrera/Downloads/flight_logs" 

# List all files and directories in the current directory
files_and_dirs = os.listdir(directory)

# Print the files only
files = [
    f for f in files_and_dirs if os.path.isfile(os.path.join(directory, f))
]

files = [f for f in files if f.endswith(".json")]

# plot_data = {}
bar_data = defaultdict(dict)
scatter_data = defaultdict(list)

for j in files:
    # Open and read the JSON file
    with open(f"{directory}/{j}", "r") as file:
        info = json.load(file)
        name = j.split(".")[0]
        # bar data
        bar_data[name]["step_count"] = info[-1].get("step_count", len(info))

        # scatter data
        if "detection_score" in info[-1].keys():
            for element in info:
                scatter_data[name].append(
                    (element["step_count"], element["detection_score"])
                )

# Create a line plot for each group
plt.figure(figsize=(10, 6))

for key, values in scatter_data.items():
    x_values = [v[0] for v in values]  # Extract x-values
    y_values = [v[1] for v in values]  # Extract y-values
    plt.plot(x_values, y_values, label=key)  # Plot line with markers


# Add labels, title, and grid
plt.xlabel("X-axis", fontsize=12)
plt.ylabel("Y-axis", fontsize=12)
plt.title("Line Plot with Legend Outside", fontsize=14)
plt.grid(True)

# Add legend outside the plot
plt.legend(
    title="Groups",
    fontsize=10,
    loc="upper left",  # Place legend in the upper left
    bbox_to_anchor=(1, 1),  # Position legend outside the plot area
)

# Adjust layout to accommodate the legend
plt.tight_layout(rect=[0, 0, 0.8, 1])

# Show the plot
plt.show()


# Extract data for plotting
filenames = list(bar_data.keys())
step_counts = [info["step_count"] for info in bar_data.values()]

# Plotting
plt.figure(figsize=(15, 6))
plt.bar(filenames, step_counts, color="skyblue")

# Customize the plot
plt.xlabel("Files", fontsize=12)
plt.ylabel("Step Count", fontsize=12)
plt.title("Step Count by File", fontsize=14)
plt.xticks(rotation=90, fontsize=8)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to fit labels

# Show the plot
plt.show()
