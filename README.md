# Dogs vs Cats Image Classification using CNNs

A deep learning project implementing a Convolutional Neural Network (CNN) to classify images of cats and dogs using TensorFlow/Keras.

## Project Overview

This project demonstrates binary image classification using deep learning techniques to distinguish between images of cats and dogs. The implementation uses a multi-layer CNN architecture with advanced regularization techniques to achieve robust classification performance.

## Dataset

- **Source**: [Dogs vs Cats dataset from Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
- **Training Images**: 20,000 images (10,000 cats, 10,000 dogs)
- **Validation Images**: 5,000 images (2,500 cats, 2,500 dogs)
- **Image Format**: RGB color images
- **License**: Available under Kaggle's dataset terms

## Data Preprocessing

### Image Preprocessing Pipeline
1. **Resizing**: All images resized to 256×256 pixels
2. **Normalization**: Pixel values normalized to range [0, 1] by dividing by 255
3. **Batch Processing**: Images processed in batches of 32
4. **Data Loading**: Automated dataset loading using `keras.utils.image_dataset_from_directory()`

### Dataset Configuration
- **Input Shape**: (256, 256, 3)
- **Batch Size**: 32
- **Shuffle**: Enabled for training data
- **Color Mode**: RGB
- **Interpolation**: Bilinear

## CNN Architecture

### Model Structure
The CNN consists of the following layers:

#### Convolutional Blocks
1. **First Block**:
   - Conv2D: 32 filters, 3×3 kernel, ReLU activation
   - BatchNormalization
   - MaxPooling2D: 2×2 pool size

2. **Second Block**:
   - Conv2D: 64 filters, 3×3 kernel, ReLU activation
   - BatchNormalization
   - MaxPooling2D: 2×2 pool size

3. **Third Block**:
   - Conv2D: 128 filters, 3×3 kernel, ReLU activation
   - BatchNormalization
   - MaxPooling2D: 2×2 pool size

#### Dense Layers
- **Flatten Layer**: Converts 2D feature maps to 1D
- **Dense Layer 1**: 128 neurons, ReLU activation, 10% Dropout
- **Dense Layer 2**: 64 neurons, ReLU activation, 10% Dropout
- **Output Layer**: 1 neuron, Sigmoid activation (binary classification)

### Model Parameters
- **Total Parameters**: 14,848,193 (56.64 MB)
- **Trainable Parameters**: 14,847,745
- **Non-trainable Parameters**: 448 (BatchNormalization parameters)

## Training Configuration

### Hyperparameters
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 2 (demonstration)
- **Batch Size**: 32

### Training Process
- **Training Data**: 20,000 images (625 batches)
- **Validation Data**: 5,000 images (157 batches)
- **Hardware**: Google Colab environment
- **Training Time**: ~2 minutes per epoch

## Results

### Training Performance
| Epoch | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
|-------|------------------|---------------|--------------------|-----------------|
| 1     | 57.43%           | 3.0140        | 53.80%             | 0.7490          |
| 2     | 68.41%           | 0.5887        | 74.78%             | 0.5090          |

### Key Observations
- **Validation Accuracy**: Achieved 74.78% after 2 epochs
- **Model Convergence**: Significant improvement from epoch 1 to 2
- **Generalization**: Validation accuracy higher than training accuracy, indicating good generalization
- **Loss Reduction**: Training loss decreased from 3.01 to 0.59

### Model Validation
- Successfully tested on external cat image from internet
- Model prediction output: `[[1.]]` (indicating correct classification)
- Demonstrates real-world applicability beyond training dataset

## Technical Implementation

### Dependencies
```python
tensorflow
keras
matplotlib
opencv-python (cv2)
kaggle
```

### File Structure
```
cnn/
├── cnn1.ipynb          # Main Jupyter notebook
├── README.md           # Project documentation
└── kaggle.json         # Kaggle API credentials
```

### Key Features
- **Batch Normalization**: Accelerates training and improves stability
- **Dropout Regularization**: Prevents overfitting (10% dropout rate)
- **Data Augmentation Ready**: Architecture supports easy integration of data augmentation
- **Scalable Design**: Can be easily extended for multi-class classification

## Usage

1. **Setup Environment**:
   ```bash
   pip install tensorflow keras matplotlib opencv-python kaggle
   ```

2. **Configure Kaggle API**:
   - Download `kaggle.json` from Kaggle account settings
   - Place in project directory

3. **Run Notebook**:
   - Open `cnn1.ipynb` in Jupyter or Google Colab
   - Execute all cells sequentially

4. **Model Training**:
   - Dataset will be automatically downloaded and extracted
   - Model will train for specified epochs
   - Training progress and accuracy plots will be displayed

## Future Improvements

### Model Enhancements
- **Extended Training**: Increase epochs for better convergence
- **Data Augmentation**: Add rotation, zoom, horizontal flip
- **Transfer Learning**: Implement pre-trained models (ResNet, VGG)
- **Hyperparameter Tuning**: Optimize learning rate, batch size

### Performance Optimization
- **Learning Rate Scheduling**: Implement adaptive learning rates
- **Early Stopping**: Prevent overfitting with validation monitoring
- **Model Checkpointing**: Save best performing models
- **Cross-Validation**: Implement k-fold validation

## License

This project is open source and available under the MIT License.

## Author

**abhijha8287** - [GitHub Profile](https://github.com/abhijha8287)

---

*This project demonstrates fundamental concepts in deep learning and computer vision, showcasing the practical application of CNNs for binary image classification tasks.*
