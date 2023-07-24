import pandas as pd
from sklearn.decomposition import PCA

# Load the features from the previous CSV file
features_df = pd.read_csv('C:/Users/91967/OneDrive/Desktop/Machinelearning/output.csv')

# Apply PCA
pca = PCA(n_components=5)  # Specify the desired number of components
pca_result = pca.fit_transform(features_df)

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2','PC3','PC4','PC5'])

# Save the PCA results to a CSV file
pca_df.to_csv('pca_output.csv', index=False)
