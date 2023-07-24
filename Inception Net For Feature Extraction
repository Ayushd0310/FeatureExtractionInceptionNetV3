import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# Define InceptionV3 model
model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')

# Define image directory and output directory
img_dir = 'C:/Users/91967/OneDrive/Desktop/Machinelearning/TrainingTumor'
output_file = 'output.csv'

# Get all image filenames in the directory
img_filenames = os.listdir(img_dir)

# Define the number of features
num_features = 2048

# Initialize an empty DataFrame to store the features
features_df = pd.DataFrame(columns=[f'feature_{i}' for i in range(num_features)])

# Loop through each image and extract features using InceptionV3
for filename in img_filenames:
    img_path = os.path.join(img_dir, filename)
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)

    # Create a DataFrame row and append it to the features DataFrame
    features_row = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(num_features)])
    features_df = features_df.append(features_row, ignore_index=True)

# Save the DataFrame to a CSV file
features_df.to_csv(output_file, index = False)
