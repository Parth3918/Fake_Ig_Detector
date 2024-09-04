# 📷 Instagram Fake Profile Detection

This project aims to detect fake profiles on Instagram using various machine learning techniques. It leverages multiple classification algorithms to predict the likelihood of a profile being fake based on specific features.

---

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 📝 Project Overview

This project focuses on building a machine learning model to identify fake Instagram profiles. The models are trained on a dataset containing features such as the number of posts, followers, following, and other relevant attributes. The primary goal is to classify profiles as either real or fake.

## 📊 Dataset

The dataset used in this project includes the following features:
- **Number of posts**
- **Number of followers**
- **Number of profiles followed**
- **Presence of external URLs**
- **Private or public profile**

The dataset is split into training and testing sets to evaluate the model's performance.

## ⚙️ Installation

To run this project, you'll need Python installed along with several key libraries. You can install the necessary packages using the following command:

pip install -r requirements.txt

## Required Packages:

-numpy
-pandas
-matplotlib
-seaborn
-scikit-learn
-keras


## 🚀 Usage
To run the project, execute the Jupyter notebook provided. The notebook will guide you through the entire process from data loading and preprocessing to model training and evaluation.

Load the dataset: The dataset is loaded and preprocessed to handle missing values and encode categorical variables.
Train the model: Various machine learning models are trained on the processed data.
Evaluate the model: The performance of the models is evaluated using metrics like accuracy and confusion matrix.


## 🤖 Models Used
The following machine learning models were implemented:

Logistic Regression
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Decision Tree Classifier
Random Forest Classifier
Naive Bayes
Neural Networks using Keras


## 📈 Results
The model performance is measured using accuracy and confusion matrix. The final section of the notebook provides a comparative analysis of all the models used.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to improve the project.

## 📄 License
This project is licensed under the MIT License.


