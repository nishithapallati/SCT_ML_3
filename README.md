# 🐶🐱 Dogs vs Cats Image Classification using SVM

## 📌 Overview

This project implements an Image Classification model to distinguish between images of dogs and cats using **Support Vector Machine (SVM)** and **Histogram of Oriented Gradients (HOG)** feature extraction.

The model is trained on the popular Kaggle Dogs vs Cats dataset and can classify new images as either a dog or a cat.

---

## 🎯 Objective

The goal of this project is to:

* Learn image preprocessing techniques.
* Extract meaningful features from images using HOG.
* Train a Support Vector Machine classifier.
* Evaluate model performance.
* Predict classes for unseen images.

---

## 📂 Dataset

Dataset used:

**Dogs vs Cats Dataset (Kaggle)**

The dataset contains thousands of labeled images belonging to two classes:

* Cat
* Dog

Dataset Structure:

```text
train/
├── cat.0.jpg
├── cat.1.jpg
├── ...
├── dog.0.jpg
├── dog.1.jpg
└── ...

This project uses the Dogs vs Cats dataset from Kaggle:

https://www.kaggle.com/c/dogs-vs-cats/data

Download the dataset and place it in:

dataset/dogs-vs-cats/train/train  
---

## 🛠 Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-Learn
* Scikit-Image
* Matplotlib
* Joblib
* tqdm

---

## 📁 Project Structure

```text
DogVsCat(SVM)
│
├── dataset/
│   └── dogs-vs-cats/
│       └── train/
│
├── images/
│   └── sample.jpg
│
├── models/
│   └── svm_model.pkl
│
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Feature Extraction

The project uses **Histogram of Oriented Gradients (HOG)** to extract important image features.

Benefits of HOG:

* Captures edge and shape information.
* Reduces dimensionality.
* Improves classification performance.
* Works effectively with SVM classifiers.

---

## 🤖 Machine Learning Model

### Support Vector Machine (SVM)

SVM is a supervised machine learning algorithm used for classification tasks.

Advantages:

* Effective for binary classification.
* Works well with HOG features.
* Produces robust decision boundaries.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/DogVsCat-SVM.git
cd DogVsCat-SVM
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Training the Model

Run:

```bash
python train.py
```

The script will:

* Load images
* Preprocess images
* Extract HOG features
* Train SVM classifier
* Evaluate accuracy
* Save the trained model

Model file generated:

```text
models/svm_model.pkl
```

---

## 🔍 Predicting New Images

Place an image inside:

```text
images/
```

Update image path in `predict.py` if necessary and run:

```bash
python predict.py
```

Example Output:

```text
Prediction: Dog
```

or

```text
Prediction: Cat
```

---

## 🎥 Demo Video
https://drive.google.com/file/d/1EYZTql58Tqei3iSC94FDAjhUICZt1YA2/view?usp=drive_link

## 📊 Model Evaluation

Evaluation metrics used:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 📈 Learning Outcomes

Through this project, I learned:

* Image preprocessing with OpenCV
* Feature extraction using HOG
* Machine Learning classification using SVM
* Model training and evaluation
* Saving and loading ML models with Joblib
* Building an end-to-end image classification pipeline

---

## 🌟 Future Improvements

* Train on the complete dataset
* Hyperparameter tuning
* Use CNN-based Deep Learning models
* Build a Flask web application for deployment
* Deploy the model on cloud platforms

---

## 👩‍💻 Author

**Nishitha Pallati**

Machine Learning Internship Project – Image Classification using Support Vector Machine (SVM)
