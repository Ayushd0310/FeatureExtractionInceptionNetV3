import pandas as pd
import numpy as np
import skfuzzy as fuzz

# Load the PCA results from the previous CSV file
pca_df = pd.read_csv('C:/Users/91967/OneDrive/Desktop/Machinelearning/pca_output.csv')

# Define membership functions for the input variables (PC1, PC2, PC3, PC4, PC5)
pc_ranges = []
for i in range(1, 6):
    pc_range = np.linspace(pca_df[f'PC{i}'].min(), pca_df[f'PC{i}'].max(), 100)
    pc_ranges.append(pc_range)

# Define membership functions for the output classes (tumor types)
tumor_types = ['Pituitary Tumor', 'Glioma Tumor', 'No Tumor', 'Meningioma Tumor']
num_classes = len(tumor_types)

# Define fuzzy membership functions for each input variable
fuzzy_memberships = []
for pc_range in pc_ranges:
    fuzzy_memberships.append(fuzz.membership.trimf(pc_range, [pc_range[0], pc_range[0], pc_range[-1]]))  # Low
    fuzzy_memberships.append(fuzz.membership.trimf(pc_range, [pc_range[0], pc_range[int(len(pc_range)/2)] , pc_range[-1]]))  # Medium
    fuzzy_memberships.append(fuzz.membership.trimf(pc_range, [pc_range[0], pc_range[-1], pc_range[-1]]))  # High

# Define the fuzzy rules
fuzzy_rules = [
    # Rule 1: If PC1 is low AND PC5 is high, then the tumor type is No Tumor
    [(0, 9), 'No Tumor'],
    # Rule 2: If PC3 is medium OR PC4 is medium, then the tumor type is Meningioma Tumor
    [(5, 7), 'Meningioma Tumor'],
    # Rule 3: If PC2 is high OR PC5 is high, then the tumor type is Glioma Tumor
    [(3, 9), 'Glioma Tumor'],
    # Rule 4: If PC1 is high AND PC2 is high AND PC4 is high, then the tumor type is Pituitary Tumor
    [(6, 3, 7), 'Pituitary Tumor']
]

# Perform fuzzy inference to classify the tumor types based on the defined rules
classification_results = []
for i in range(len(pca_df)):
    pc_values = [pca_df[f'PC{j+1}'][i] for j in range(5)]
    fuzzy_degrees = []
    for rule in fuzzy_rules:
        fuzzy_degree = min([fuzz.interp_membership(pc_ranges[j // 3], fuzzy_memberships[rule[j][0]], pc_values[j])
                            for j in range(len(rule)-1)])
        fuzzy_degrees.append(fuzzy_degree)
    classification_results.append(fuzzy_rules[np.argmax(fuzzy_degrees)][1])


# Create a DataFrame for the classification results
classification_df = pd.DataFrame({'PC1': pca_df['PC1'], 'PC2': pca_df['PC2'], 'PC3': pca_df['PC3'], 'PC4': pca_df['PC4'], 'PC5': pca_df['PC5'], 'Classification': classification_results})

# Save the classification results to a CSV file
classification_df.to_csv('fuzzy_classification_output.csv', index=False)
