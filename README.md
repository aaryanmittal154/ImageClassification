# Image Classification with CIFAR-10

This project performs image classification on the CIFAR-10 dataset using a Convolutional Neural Network (CNN) implemented with TensorFlow and Keras.

## Project Structure

- `image_classification.py`: Main script to load, preprocess, and train the CNN model on the CIFAR-10 dataset.

## Requirements

- Python 3.6+
- Required libraries: `tensorflow`, `keras`, `numpy`, `matplotlib`

## Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/aaryanmittal/image-classification-cifar10.git
    cd image-classification-cifar10
    ```

2. **Install the required libraries**:
    ```bash
    pip install tensorflow keras numpy matplotlib
    ```

## Usage

1. **Run the image classification script**:
    ```bash
    python image_classification.py
    ```

2. **Output**:
    - The script will load and preprocess the CIFAR-10 dataset, build and train a CNN model, evaluate its performance, and display training results.

## Explanation of the Script

- **Import Libraries**: The script imports necessary libraries for data manipulation, image processing, and deep learning.
- **Load Dataset**: The CIFAR-10 dataset is loaded and normalized.
- **Data Visualization**: Displays the first 9 images from the training set with their corresponding class names.
- **Model Building**: Builds a Convolutional Neural Network (CNN) using Keras.
- **Model Training**: Trains the CNN model using the training dataset.
- **Model Evaluation**: Evaluates the model's performance on the test dataset and prints the accuracy.
- **Visualization**: Plots the training and validation accuracy and loss over epochs.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. The class names are:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The CIFAR-10 dataset was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
- Special thanks to the creators of TensorFlow and Keras for their powerful deep learning libraries.
