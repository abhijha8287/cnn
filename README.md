# Dogs vs Cats CNN

A convolutional neural network in TensorFlow/Keras that classifies images as cat or dog.

## Why this exists
Practice building a CNN from scratch for binary image classification.

## What it does
- Loads the Kaggle Dogs vs Cats data (20,000 training and 5,000 validation images) with `image_dataset_from_directory`
- Resizes images to 256 x 256 and scales pixels to [0, 1]
- Three Conv2D blocks (32, 64, 128 filters) with batch normalisation and max pooling
- Two dense layers (128, 64) with 10% dropout and a sigmoid output; 14.8M parameters
- Trains with Adam and binary cross-entropy, and plots accuracy and loss

## Project structure
```
cnn/
└── cnn1.ipynb  Data loading, model, training, plots
```

## Getting started
Requirements: Python 3.10+, Jupyter, a Kaggle API token. A GPU is recommended.
```bash
git clone https://github.com/abhijha8287/cnn.git
cd cnn
pip install jupyter tensorflow matplotlib kaggle
kaggle datasets download -d salader/dogs-vs-cats && unzip dogs-vs-cats.zip
jupyter notebook cnn1.ipynb
```

## Tech stack
Python, TensorFlow/Keras, Matplotlib, Jupyter

## Notes
Trained for 2 epochs as a demonstration. Dataset: [salader/dogs-vs-cats](https://www.kaggle.com/datasets/salader/dogs-vs-cats).
