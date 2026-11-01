Plant Disease Detection System

This is my deep learning project for plant disease detection. I trained a Vision Transformer model to check bean leaf images. It tells if leaf is healthy or has disease.

Problem Statement
Plant diseases cause big loss for farmers. This project helps to identify bean leaf disease by just uploading a leaf picture.

dataset
I used `AI-Lab-Makerere/beans' dataset from Hugging Face. It has 3 types:
 Angular Leaf Spot
 Bean Rust
 Healthy

model training
pretrained model used: `google/vit-base-patch16-224`
fine-tuned using PyTorch in Google Colab.
fine-tuned model is saved on hugging face: https://huggingface.co/shahab1/vit-beans-plant-disease

deployment
The app is made with Streamlit and running on Streamlit cloud

Repository Files:
 app.py streamlit python file for UI
 requirements.txt List of python packages

