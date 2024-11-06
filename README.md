# ODOMETER CLASSIFICATION USING TRODO DATASET

## Project Overview

The "trodo-odometer-classification" project aims to classify odometer types and extract mileage information using the TRODO dataset. The project utilizes various machine learning techniques, including K-Nearest Neighbors (KNN), Random Forest Classifier, and Convolutional Neural Networks (CNN) to accurately classify odometer images as either analog or digital.

The entire workflow is encapsulated in the Jupyter Notebook `./odometer_classification_trodo.ipynb`, which provides step-by-step instructions, code, and outputs for replicating the study.

## Usage

To replicate the project, follow these steps:

### 1. Download and Extract the Dataset

You can access the TRODO dataset from the following link: [TRODO Dataset](https://data.mendeley.com/datasets/6y8m379mkt/2)

Mount the Google Drive to access the dataset:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Explore the Dataset

Import necessary libraries and set the dataset folder path:

```python
import os
import json
import cv2

dataset_folder_path = '/content/drive/MyDrive/trodo-v01'
groundtruth_file_path = os.path.join(dataset_folder_path, 'ground truth', 'groundtruth.json')
annotations_file_path = os.path.join(dataset_folder_path, 'pascal voc 1.1', 'Annotations')
images_file_path = os.path.join(dataset_folder_path, 'images')
```

Open and inspect the `groundtruth.json` file:

```python
with open(groundtruth_file_path, 'r') as f:
    groundtruth_data = json.load(f)

groundtruth_data['odometers'][0].keys()
```

### 3. Data Preprocessing

Install the `tqdm` library for tracking the preprocessing progress. 

Preprocess the images by extracting and resizing the odometer part:

```python
import os
import cv2
import xml.etree.ElementTree as ET
```

Convert string labels to numeric labels using `LabelEncoder`:

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
```

Save and load the preprocessed data to avoid re-running preprocessing every time:

```python
import pickle
```

Split the dataset into training and testing sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4. Train and Evaluate the Models

#### KNeighborsClassifier

Train the K-Nearest Neighbors Classifier and find the best parameters using GridSearch:

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5, 6]}]
```

Evaluate the model:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```

#### Random Forest Classifier

Train the Random Forest Classifier:

```python
from sklearn.ensemble import RandomForestClassifier
```

Evaluate the model:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
```

#### CNN

Train a Convolutional Neural Network with Early Stopping:

```python
from tensorflow.keras.models import load_model
```

Load the best model weights and summarize the model:

```python
model.load_weights('best_model.h5')
model.summary()
```

### Dependencies

- Python
- Jupyter Notebook
- Google Colab (for mounting drive)
- OpenCV
- NumPy
- scikit-learn
- TensorFlow
- tqdm
- matplotlib
- seaborn
- pickle

Ensure you have all required libraries installed to run the notebook successfully.
