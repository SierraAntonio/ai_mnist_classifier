# 🧠 Handwritten Digit Classification with CNN

This project implements an end-to-end **Convolutional Neural Network (CNN)** for handwritten digit recognition using the **MNIST dataset**. The model is trained, evaluated, and tested on real images using **TensorFlow and Keras**, achieving approximately **99% accuracy** on the test set.

## 🚀 Project Overview
The goal of this project is to demonstrate the complete machine learning workflow:
- Data loading and preprocessing
- CNN model design and training
- Model evaluation and performance analysis
- Inference on real-world handwritten digit images

This project serves as a foundational example of computer vision and deep learning applied to image classification.

## 🧠 Model Architecture
The CNN model consists of:
- Input layer for 28x28 grayscale images
- Convolutional and pooling layers for feature extraction
- Fully connected (Dense) layers for classification
- Softmax output layer for multi-class prediction (digits 0–9)

## 📊 Results
- **Test Accuracy:** ~99%
- **Dataset:** MNIST (60,000 training images, 10,000 test images)
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam

## 🛠️ Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Pillow  
- Computer Vision  
- Convolutional Neural Networks (CNN)

## 📂 Project Structure
ai-mnist-cnn-classifier/
│── models/
│── train.py
│── evaluate.py
│── requirements.txt
│── README.md

bash

## ⚙️ Setup & Installation

1. Clone the repository:
'''bash
git clone https://github.com/SierraAntonio/ai_mnist_classifier.git
cd ai-mnist-cnn-classifier
Create and activate a virtual environment:

bash

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
Install dependencies:
bash
pip install -r requirements.txt
▶️ Usage
Train the model
bash
python train.py
Evaluate the model
bash
python evaluate.py
The trained model will be saved locally and reused for inference.

🧪 Example Prediction
The model can predict handwritten digits from custom images after preprocessing (grayscale, resizing, normalization).

📌 Future Improvements
Deploy the model as a REST API using FastAPI

Add Docker support

Extend the model to support other datasets

Integrate cloud deployment (AWS)

👤 Author
Antonio Martinez
B.Sc. in Robotics and Telecommunications Engineering currently pursing a Master in Artificial Intelligence.
Interested in AI, Machine Learning, Robotics, and Cloud Technologies
