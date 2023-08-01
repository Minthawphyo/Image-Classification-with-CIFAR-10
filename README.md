# Image Classification with CIFAR-10 using CNN and MobileNet (Transfer Learning)

This repository contains code for image classification on the CIFAR-10 dataset using Convolutional Neural Networks (CNN) and MobileNet (Transfer Learning) models implemented in TensorFlow/Keras.

## Description

Image classification is a fundamental task in computer vision, and the CIFAR-10 dataset serves as a standard benchmark for evaluating image classification algorithms. The dataset contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. The task is to classify these images into one of the following categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

In this project, we explore two different approaches to tackle the image classification problem. First, we implement a custom CNN architecture, which consists of multiple convolutional and dense layers, to learn and extract relevant features from the images. We train this CNN model from scratch on the CIFAR-10 training dataset.

Next, we utilize the power of transfer learning by using the MobileNet architecture, a lightweight deep learning model, which is pretrained on the large-scale ImageNet dataset. We fine-tune the last few layers of MobileNet to adapt it to the CIFAR-10 classification task. This approach allows us to leverage the prelearned features and patterns from ImageNet, potentially leading to faster convergence and better performance on CIFAR-10.

## Dependencies

- TensorFlow (>=2.0)
- NumPy
- Matplotlib
- OpenCV (cv2)

To install the required dependencies, run the following command:

pip install -r requirements.txt


## Dataset

The CIFAR-10 dataset is automatically loaded using the TensorFlow/Keras `cifar10.load_data()` function. We preprocess the data by normalizing the pixel values to a range between 0 and 1 to ensure numerical stability during training.

## CNN Model

The custom CNN model architecture is implemented using TensorFlow/Keras. It comprises multiple Convolutional, BatchNormalization, MaxPooling, and Dense layers. The model is compiled with the 'adam' optimizer and 'sparse_categorical_crossentropy' loss function.

## MobileNet Model (Transfer Learning)

The MobileNet model is a pre-trained deep learning architecture designed for mobile and embedded vision applications. We use the MobileNet model with its weights trained on the ImageNet dataset. The last few layers are fine-tuned for our CIFAR-10 classification task.

## Training

Both the CNN and MobileNet models are trained on the CIFAR-10 training dataset. We monitor the training progress over 20 epochs and record the validation accuracy. After training, we evaluate the models' performance on the CIFAR-10 test dataset.

## Results

The training and validation accuracy and loss plots are displayed for both the CNN and MobileNet models. Additionally, we test the models by predicting a sample image from the test dataset and displaying the predicted and original labels.

## Usage

1. Clone the repository to your local machine:

git clone https://github.com/your_username/Image_Classification_CIFAR-10.git
cd Image_Classification_CIFAR-10


2. Install the required dependencies:

pip install -r requirements.txt


3. Open the Jupyter notebook `Image_Classification_CIFAR-10.ipynb` using Jupyter Notebook or JupyterLab.

4. Follow the instructions and run the code cells in the notebook to preprocess the data, implement and train the CNN and MobileNet models, and evaluate the model's performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The CIFAR-10 dataset is taken from the TensorFlow/Keras dataset library.
- The MobileNet model is part of the TensorFlow/Keras application models.

Feel free to use and modify the code for your own image classification tasks!

