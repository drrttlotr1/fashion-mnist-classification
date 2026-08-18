# Fashion-MNIST Image Classification

## 📌 Project Overview

This project focuses on **image classification using deep learning** with the Fashion-MNIST dataset.

The main goal was to build and evaluate neural network models capable of classifying grayscale images of clothing items into 10 categories. The project includes experiments with a **fully connected neural network, a Convolutional Neural Network (CNN), and transfer learning using VGG16**.

The project demonstrates the complete deep learning workflow: data preparation, model development, training, evaluation, and comparison of different architectures.

## 🗂️ Dataset

**Fashion-MNIST** is a dataset of 70,000 grayscale images of fashion items:

* 60,000 training images
* 10,000 test images
* Image size: 28 × 28 pixels
* 10 classes

The classes include T-shirts/tops, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and ankle boots.

## 🧠 Models

### Fully Connected Neural Network

A dense neural network was developed as a baseline model for image classification.

**Test accuracy:** ~89%

### Convolutional Neural Network

A CNN was implemented to better capture spatial patterns and visual features in the images.

**Test accuracy:** ~90.6%

The training process included:

* Early Stopping
* ReduceLROnPlateau
* Validation monitoring
* Model performance evaluation

### VGG16 Transfer Learning

The project also explores the use of the **VGG16 architecture** for image classification and demonstrates the challenges of applying a pretrained architecture to the Fashion-MNIST dataset.

## 📊 Results

| Model                | Test Accuracy |
| -------------------- | ------------: |
| Dense Neural Network |        ~89.0% |
| CNN                  |    **~90.6%** |
| VGG16                |  Experimental |

The CNN provided the best result among the successfully trained models, improving classification performance compared with the baseline dense neural network.

## 🛠️ Technologies

* **Python**
* **TensorFlow**
* **Keras**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**
* **Streamlit**

## 📁 Project Structure

```text
goit-ds-hw-16-main/
│
├── Hw13.ipynb          # Data analysis, model development and experiments
├── app.py              # Application for model demonstration
├── requirements.txt    # Project dependencies
├── cnn_history.npy     # CNN training history
├── vgg_history.npy     # VGG16 training history
├── images.jpg          # Project image
└── name1_man_29.jpg    # Image used for model demonstration
```

## 🎯 Key Takeaways

This project provided practical experience with:

* Preparing image data for deep learning
* Building neural networks with TensorFlow/Keras
* Designing and training CNN architectures
* Applying regularization and training callbacks
* Evaluating classification models
* Comparing different neural network architectures
* Exploring transfer learning with VGG16
* Deploying a trained model through a simple application

## 👩‍💻 Author

**Svitlana Melnyk**

Python / Data Science Developer

[GitHub](https://github.com/drrttlotr1)
