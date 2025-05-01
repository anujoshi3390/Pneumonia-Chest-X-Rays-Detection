Pneumonia is an infection that affects one or both lungs by causing the air sacs, or alveoli, of the lungs to fill up with fluid or pus. Traditionally, pneumonia detection hinges on the examination of chest X-ray radiographs, a labor-intensive process conducted by highly skilled specialists. This method often results in discordant interpretations among radiologists. Leveraging the power of deep learning techniques (convolutional neural networks), we have developed a computational approach for the detection of pneumonia regions.

![Unknown-1](https://github.com/user-attachments/assets/39c37361-9ac9-4633-8bd0-acd812d97a69)


I developed an AI system for pneumonia detection using Convolutional Neural Networks (CNNs), implemented in Python with TensorFlow and the Keras API.

The model was trained on the Chest X-Ray Images (Pneumonia) dataset from Kaggle, which includes chest X-rays categorized as Normal or Pneumonia. The dataset is organized into train, test, and val directories and consists of images from pediatric patients (ages 1–5) at Guangzhou Women and Children's Medical Center. Low-quality scans were removed through a quality control process, and expert physicians verified the diagnoses to ensure dataset reliability.

To enhance the training data, we used ImageDataGenerator for real-time data augmentation, applying transformations such as scaling, rotation, flipping, zooming, and shifting. Images were normalized by rescaling (1.0/255) and resized to 64x64 pixels. This helped compensate for the relatively small dataset size (~6000 images).

The CNN architecture includes:

Input layer with dropout to prevent overfitting
First convolutional block: 32 filters (3x3), tanh activation
Second block: Max-pooling (2x2), 64 filters, tanh activation
Third block: Max-pooling, 64 filters, tanh activation
Dropout layers after each convolutional block
Dense output layer for binary classification
Technologies used:
Python
TensorFlow
Keras
NumPy
