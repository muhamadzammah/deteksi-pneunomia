This project is a pneumonia detection system based on chest X-ray images that combines deep learning with traditional machine learning. The system is built using Flask as the web framework, MobileNetV2 as the visual feature extractor, and KNN (K-Nearest Neighbors) as the classification model.

The application provides two main functionalities:

1. Web Application for uploading X-ray images, viewing prediction results, severity levels, medical recommendations, and patient history.

2. REST API for integration needs, such as with Flutter mobile applications.

Main Workflow
1. Image Upload and Preprocessing

When a user uploads an X-ray image, the system saves the file and generates a series of preprocessing stages, including:

RGB Conversion

Histogram Equalization

Otsu Thresholding

Gaussian Blur

Morphological Opening

Canny Edge Detection

All steps are displayed to the user for transparency regarding how the image is processed before analysis.

2. Feature Extraction with MobileNetV2

MobileNetV2 is used as the feature extractor without its top classification layers. Each image is converted into a feature vector through:

input preprocessing

MobileNetV2 backbone inference

global average pooling

The resulting feature vector is then classified using a KNN model.

3. Pneumonia Classification Using KNN

The KNN model is optimized using GridSearchCV to find the best combination of:

number of neighbors (k)

weighting method

distance metric

The best model is saved and used in the Flask application.

Prediction outputs include:

Label (Normal or Pneumonia)

Confidence score

Severity category (Mild / Moderate / Severe)

Medical recommendations based on severity

4. Patient Data Storage

All prediction results are stored in the database using SQLAlchemy. Stored data includes:

Patient identity

X-ray file path

Prediction results

Confidence

Severity

Admins can view all records via the /admin page and delete data when needed.

5. API Endpoint

The system also includes a /predict endpoint that processes images through an API, enabling integration with mobile apps or other external systems.

Model Training with a Separate Script

A dedicated training script is provided, which:

Loads the dataset from dataset/chest_xray/train

Splits it into train, validation, and test sets

Extracts features using MobileNetV2

Performs scaling and label encoding

Runs GridSearch to obtain the best KNN configuration

Saves the model, scaler, and label encoder

The generated model is then used by the main Flask application for prediction.