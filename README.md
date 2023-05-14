# Pneumonia-Classification-on-TPU


This repository contains code for a deep learning model that classifies chest X-ray images into two classes: normal and pneumonia. The model is built using the TensorFlow framework and is optimized for running on Google Cloud's TPU (Tensor Processing Unit).

## Dataset
The model is trained on the Chest X-Ray Images (Pneumonia) dataset, which is publicly available on Kaggle. The dataset contains a total of 5,856 chest X-ray images, with 3,680 images showing signs of pneumonia and 1,920 images labeled as normal.

## Model Architecture
The model is built using a convolutional neural network (CNN) with the following architecture:

1. Input Layer (224x224x3)
2. Convolutional Layer (32 filters, kernel size of 3x3, ReLU activation)
3. Max Pooling Layer (2x2)
4. Convolutional Layer (64 filters, kernel size of 3x3, ReLU activation)
5. Max Pooling Layer (2x2)
6. Convolutional Layer (128 filters, kernel size of 3x3, ReLU activation)
7. Max Pooling Layer (2x2)
8. Convolutional Layer (256 filters, kernel size of 3x3, ReLU activation)
9. Max Pooling Layer (2x2)
10. Flatten Layer
11. Dense Layer (512 units, ReLU activation)
12. Dropout Layer (0.5 probability)
13. Output Layer (2 units, Softmax activation)

## Training on TPU
The model is trained on a Google Cloud TPU to take advantage of its superior performance and speed. The code includes instructions for setting up and connecting to a TPU instance, as well as for configuring the TPU for training.

## Requirements
To run this code, you will need the following dependencies:

1. TensorFlow 2.x
2. TensorFlow Cloud
3. Google Cloud SDK
4. Kaggle API

## Usage
### To run the code, first clone this repository and navigate to the directory:

git clone https://github.com/username/Pneumonia_Classification_on_TPU.git cd Pneumonia_Classification_on_TPU


### Next, install the necessary dependencies using pip:

pip install -r requirements.txt


Before training the model, you will need to set up your Google Cloud and Kaggle credentials. Follow the instructions in the config_template.py file to create a new config.py file with your credentials.


### Once your credentials are set up, you can start training the model by running the following command:

python train.py


### This will start the training process on the TPU. Once training is complete, you can evaluate the model on the test set by running:

python evaluate.py


## Result:

![image](https://github.com/PurnaChandar26/Pneumonia-Classification-on-TPU/assets/97793147/28d5ac0d-a97d-4e8f-b77d-a45817c93fa9)
## Conclusion

We hope that this repository will be useful for those interested in building and training deep learning models for medical image classification. If you have any questions or feedback, please feel free to open an issue or submit a pull request.


