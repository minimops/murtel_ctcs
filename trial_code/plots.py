import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
filename = '../feature_extract/combined_feat_labels.csv'  # Replace with your actual filename
data = pd.read_csv(filename)

# Drop unnecessary columns
columns_to_drop = ['image', 'label', 'tri_label']
data = data.drop(columns=columns_to_drop)

# Extract features and labels
features = [col for col in data.columns if col != 'bin_label']
labels = data['bin_label'].unique()

# Grid size for pair plot
num_features = len(features)

# Create larger figure
fig, axes = plt.subplots(num_features, num_features, figsize=(25, 25), constrained_layout=True)

label_colors = {
    0: 'green',
    1: 'red'
}

for i, feature1 in enumerate(features):
    for j, feature2 in enumerate(features):
        ax = axes[i, j]

        if i <= j:
            # Leave diagonal empty
            ax.axis('off')
        else:
            # Plot scatter plots for off-diagonal elements with smaller dots
            for label in labels:
                subset = data[data['bin_label'] == label]
                ax.scatter(subset[feature2], subset[feature1], alpha=0.2, s=0.1,
                           c=label_colors[label])  # Set smaller dot size with `s`

        # Add feature labels to the first column and first row
        if j == 0:
            ax.set_ylabel(feature1, fontsize=14)  # Larger font for axis labels
        else:
            ax.set_ylabel('')

        if i == num_features - 1:
            ax.set_xlabel(feature2, fontsize=14)  # Larger font for axis labels
        else:
            ax.set_xlabel('')

        ax.grid(True)

# Save and display the plot
plt.savefig('feature_plot.png', dpi=300)  # Save with higher resolution
