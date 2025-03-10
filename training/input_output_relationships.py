import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif

# Load the data
df = pd.read_csv('mushroom_data.csv', header=0) 

feature_names = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 
    'stalk-surface-below-ring', 'stalk-color-above-ring', 
    'stalk-color-below-ring', 'veil-type', 'veil-color', 
    'ring-number', 'ring-type', 'spore-print-color', 
    'population', 'habitat'
]

# Encode the categorical columns using LabelEncoder
le = LabelEncoder() 
label_mappings = {}

for column in df.columns:
    if df[column].dtype == 'object': 
        df[column] = le.fit_transform(df[column])
        label_mappings[column] = dict(zip(le.classes_, range(len(le.classes_))))

# Split the data into features and target
features = df.iloc[:, :-1] # all columns except the last one
target = df.iloc[:, -1] # last column

# Explore the features and target relationship using line plots
x = np.arange(1, 31, 1)
plt.figure(figsize=(12, 8))

# Plot each feature against the target using the first 30 data points
for i, feature in enumerate(features.columns):
    plt.subplot(6, 4, i + 1) # Create subplots in a grid of 6 rows and 4 columns
    plt.plot(x, features[feature][:30], color='blue') 
    plt.plot(x, target[:30], color="#ff21d7") 
    plt.title(feature.replace('-', ' ').title(), fontsize=8)
# Add a title for the overall plot
plt.suptitle('Mushroom Features(blue) vs Poisonous(pink)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

# Functions to add 10 more colors to the tab20 color palette, to be used in the parallel plot
def interpolate_color(color1, color2, fraction, alpha=1.0):
    interpolated_color = tuple(c1 + (c2 - c1) * fraction for c1, c2 in zip(color1, color2))
    return (*interpolated_color[:3], alpha) 

def create_tab30():
    tab20 = plt.cm.get_cmap('tab20')
    tab20_colors = [tab20(i) for i in range(20)]
    tab30_colors = []

    for i in range(0, len(tab20_colors) - 1, 2): 
        color1 = tab20_colors[i]
        color2 = tab20_colors[i + 1]
        tab30_colors.append(color1) 
        tab30_colors.append(interpolate_color(color1, color2, 0.5, alpha=1.0))  
        tab30_colors.append(color2)  

    if len(tab20_colors) % 2 != 0:
        tab30_colors.append(tab20_colors[-1])
    return tab30_colors

tab30 = create_tab30()

# Explore the features and target relationship using parallel plot
plt.figure(figsize=(12, 8))
# Plot the first 30 data points for all features and target
for i in range(30): 
    plt.plot(df.columns, df.iloc[i, :], marker='o', label=f'Observation {str(i + 1)}', color=tab30[i])
    
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.xlabel('Features')
plt.ylabel('Normalized Values')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.title('Parallel Plot')
plt.tight_layout()
plt.show()

# Explore the features and target relationship using parallel coordinates plot
fig = px.parallel_coordinates(
    df.iloc[:30],  # Use the first 30 data points
    color='poisonous', # Color the lines based on the target
    dimensions=features.columns.tolist(), # Use all features
    color_continuous_scale=px.colors.diverging.Tealrose, 
    title='Parallel Coordinate Plot for Mushroom Dataset'
    )
# Save the plot as an HTML file
fig.write_html('parallel_coordinates_plot.html')

# Explore the features and target relationship using bar plots
features_to_plot = df.columns[:-1]
num_features = len(features_to_plot)
color_palette = {0: '#25ff8f', 1: '#ff21d7'} 
plt.figure(figsize=(12, 8))

# Loop through each feature for plotting
for i, column in enumerate(features_to_plot):
    plt.subplot(6, 4, i + 1)  # Create subplots in a grid of 6 rows and 4 columns
    sns.countplot(data=df, x=column, hue='poisonous', palette=color_palette, legend=False)
    # Since the data has been encoded, need to use label mappings to set the labels
    plt.xticks(ticks=np.arange(len(label_mappings[column])), 
               labels=list(label_mappings[column].keys()), 
               fontsize=6)
    plt.xlabel('')    
    plt.ylabel('Count', fontsize=8)    
    plt.title(column.replace('-', ' ').title(), fontsize=8)    
# Add a title for the overall plot
plt.suptitle('Mushroom Features vs Poisonous Status (Edible-Green, Poisonous-Pink)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

# Explore the features and target relationship using feature selection
# Remove constant columns from the features as F-test cannot be calculated for them
constant_columns = [i for i in range(features.shape[1]) if features.iloc[:, i].nunique() == 1]
print(f"Removed {len(constant_columns)} constant columns: {features.columns[constant_columns].tolist()}")
features = features.drop(features.columns[constant_columns], axis=1)
feature_names = [name for i, name in enumerate(feature_names) if i not in constant_columns]

# Create f_classif object to calculate the F-values and p-values
f_value, p_value = f_classif(features, target)

# Print the F-value and p-value for each feature
for feature, f_val, p_val in zip(feature_names, f_value, p_value):
    print(f"Feature: {feature}, F-value: {f_val:.2f}, p-value: {p_val}")

# Create a bar chart for visualizing the F-values
plt.figure(figsize=(12, 8))
bars = plt.bar(x=feature_names, height=f_value, color='tomato')
# Add the F-values as text on top of the bars
for bar, f_val in zip(bars, f_value):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f'{f_val:.2f}', ha='center', va='bottom', fontsize=8)      
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.ylabel('F-value')
plt.title('F-value Comparison')
plt.tight_layout()
plt.show()

# Create SelectKBest object to select the 8 best features
selector = SelectKBest(f_classif, k=8)

# Perfrom the selection of k best features
features_new = selector.fit_transform(features, target)
selected_indices = selector.get_support(indices=True)
selected_features = [
    (feature_names[i], f_value[i], p_value[i]) for i in selected_indices
]
sorted_selected_features = sorted(selected_features, key=lambda x: x[1], reverse=True)

print("Top 8 Selected Features:")
for feature, f_val, p_val in sorted_selected_features:
    print(f"Feature: {feature}, F-value: {f_val:.2f}, p-value: {p_val}")