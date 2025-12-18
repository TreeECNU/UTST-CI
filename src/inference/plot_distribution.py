import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data from provided statistics
data = {
    'Vocabulary Difficulty': ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'],
    'elementary': [0.6381, 0.1891, 0.0847, 0.0518, 0.0063, 0.0299],
    'middle': [0.5848, 0.1989, 0.1065, 0.0672, 0.0082, 0.0345],
    'high': [0.5245, 0.2100, 0.1315, 0.0852, 0.0105, 0.0383],
    'college': [0.4497, 0.2211, 0.1614, 0.1092, 0.0137, 0.0449]
}

# Create DataFrame
df = pd.DataFrame(data)

# Set up the plot
plt.figure(figsize=(12, 6))

# Parameters for grouped bar plot
readability_levels = ['elementary', 'middle', 'high', 'college']
n_levels = len(readability_levels)
n_categories = len(df['Vocabulary Difficulty'])
bar_width = 0.2
index = np.arange(n_categories)

# Plot bars for each readability level
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, (level, color) in enumerate(zip(readability_levels, colors)):
    bars = plt.bar(
        index + i * bar_width,
        df[level],
        bar_width,
        label=level.capitalize(),
        color=color,
        alpha=0.8
    )
    # Add value annotations on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=8
        )

# Customize labels and layout
plt.xlabel('Vocabulary Difficulty Category', fontsize=12)
plt.ylabel('Proportion', fontsize=12)
plt.title('Vocabulary Difficulty Distribution by Readability Level', fontsize=14, pad=20)
plt.xticks(index + bar_width * (n_levels - 1) / 2, df['Vocabulary Difficulty'])
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Readability Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the plot
plt.savefig('vocabulary_difficulty_bars.png')
plt.close()