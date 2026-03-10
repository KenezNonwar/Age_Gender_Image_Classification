# Age & Gender Prediction using CNN

A Deep Learning project that predicts a person's **age and gender from facial images** using a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**.

This project demonstrates a full ML workflow including preprocessing, visualization, training, and prediction.

---

## Project Overview

The model predicts:

- Age (Regression)
- Gender (Binary Classification)

The system learns facial patterns from images in the **UTKFace dataset**.

---

## Dataset

Dataset: **UTKFace**

The dataset contains around **20,000 facial images** with annotations for:

- Age
- Gender
- Race

Filename format:

age_gender_race_timestamp.jpg

Example:

25_0_2_20170116174525125.jpg

Where:
- 25 = Age
- 0 = Male
- 1 = Female

---

## Project Structure

Age-Gender-Prediction-CNN
│
├── main.ipynb
├── requirements.txt
├── age_gender_model.h5
└── README.md

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Scikit-learn

---

## Machine Learning Pipeline

1. Dataset loading
2. Data preprocessing
3. Data visualization
4. Train/test split
5. CNN model training
6. Prediction

---

## Example Prediction

Predicted Age: 28  
Predicted Gender: Male

---

## Installation

git clone https://github.com/KenezNowar/age-gender-prediction-cnn.git

cd age-gender-prediction-cnn

pip install -r requirements.txt

---

## Run

jupyter notebook main.ipynb

or run in Google Colab.

---

## Future Improvements

- Transfer learning (VGG16 / ResNet)
- Better evaluation metrics
- Web app using Flask or Streamlit
- Model deployment

---

## Author

Ken Nonwar  
Aspiring AI Engineer / Machine Learning Developer
