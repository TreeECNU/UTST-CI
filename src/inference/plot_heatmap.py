import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data from the provided statistics
data = {
    "intended_label": ["college", "high", "middle", "elementary"],
    "category_counts": [
        {"elementary": 30, "middle": 26, "high": 576, "college": 3368},
        {"elementary": 13, "middle": 641, "high": 2509, "college": 837},
        {"elementary": 255, "middle": 2553, "high": 1135, "college": 57},
        {"elementary": 2143, "middle": 1745, "high": 110, "college": 2}
    ]
}

# Define all possible categories
categories = ["elementary", "middle", "high", "college"]

# Create a DataFrame for the heatmap, filling missing categories with 0
heatmap_data = pd.DataFrame(
    [[counts.get(cat, 0) for cat in categories] for counts in data["category_counts"]],
    index=data["intended_label"],
    columns=categories
)

# Set up the plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    cbar_kws={'label': 'Count'},
    square=True,
    annot_kws={"size": 16, "weight": "bold"}
)

# Customize labels
plt.title("Heatmap of Predicted Sentiment Categories by Intended Label", fontsize=20, pad=20)
plt.xlabel("Predicted Category", fontsize=20)
plt.ylabel("Intended Label", fontsize=20)

# 修改 x 轴和 y 轴类别标签的字体大小
plt.xticks(fontsize=14)  # x 轴类别标签字体大小
plt.yticks(fontsize=14)  # y 轴类别标签字体大小

# Save the plot
plt.savefig('readability_heatmap.png')
plt.close()