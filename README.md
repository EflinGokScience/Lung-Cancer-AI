# 🫁 Lung Cancer AI Detection System

An AI-powered lung cancer CT scan classification system developed using **EfficientNetB0** and **Grad-CAM**.

## Overview

This project classifies lung CT scan images into four categories:

* Adenocarcinoma
* Large Cell Carcinoma
* Squamous Cell Carcinoma
* Normal

The system also explains its predictions using **Grad-CAM**, allowing users to visualize which regions of the CT image influenced the model's decision.

> **Research and educational use only. This project is not intended for clinical diagnosis.**

---

## Features

* EfficientNetB0 deep learning model
* Four-class CT scan classification
* Grad-CAM explainability
* Professional AI demo interface
* ROC/AUC evaluation
* Error analysis
* Confusion matrix
* Classification report

---

## Model Performance

| Metric             | Result     |
| ------------------ | ---------- |
| Test Accuracy      | **90.21%** |
| Adenocarcinoma AUC | **0.973**  |
| Large Cell AUC     | **0.983**  |
| Normal AUC         | **0.998**  |
| Squamous AUC       | **0.969**  |

---

## Technologies

* Python
* TensorFlow / Keras
* EfficientNetB0
* OpenCV
* NumPy
* Matplotlib
* Gradio

---

## Future Work

* Clinical validation using independent hospital datasets
* Tumor segmentation
* Tumor size estimation
* Integration of additional clinical metadata

---

## Author

Developed by **Eflin Gok** as a Year 10 AI research project.
# Lung-Cancer-AI
AI-powered lung cancer CT scan classification using EfficientNetB0 and Grad-CAM.
