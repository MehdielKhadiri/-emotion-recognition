# Emotion-recognition
This repository contains Python scripts for an application for emotion recognition. The application is developed with TensorFlow and Keras, using Convolutional Neural Networks (CNNs) and webcam image capture in real-time. The model is trained and evaluated using a dataset of grayscale images, each of which is classified as one of seven emotions.
## Setup and Requirements

The application requires the following software and libraries:

- Python 3.6 or later
- TensorFlow 2.x
- Keras
- OpenCV
- h5py
- CUDA (optional, for GPU support)

To install the necessary libraries, you can use pip:

```bash
pip install tensorflow keras opencv-python h5py
```

Please note: If you plan to use a CUDA-enabled GPU, make sure to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) according to the instructions provided by NVIDIA.

## Usage

After installing the dependencies, you can run the training script:

```bash
python webcamcnn.py
```

After training, you can run the testing script:

```bash
python webcamtest.py
```

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.
