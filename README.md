# Pneumonia Classifier

Project for classification of chest X-ray images for pneumonia detection using deep learning.

## 📋 Project Description

This project is an end-to-end solution for classifying lung X-ray images into two categories:
- **Normal** - normal lung condition
- **Pneumonia** - presence of pneumonia

The solution includes:
1. Training a deep neural network (ResNet18) on a dataset of X-ray images
2. Module for predicting on new images
3. REST API using FastAPI for model deployment
4. Docker containers for easy deployment
5. Experiment tracking with MLflow
6. Deployment to Azure Container Instances

## 🗂️ Project Structure

```
pneumonia-classifier/
├── pneumonia_classifier/          # Main package
│   ├── api/                      # API layer
│   │   ├── main.py               # FastAPI application
│   │   └── routes.py             # API endpoints
│   ├── data/                     # Data handling
│   │   └── dataset.py            # Datasets and transformations
│   ├── models/                   # Models
│   │   └── resnet.py             # ResNet model implementation
│   ├── config.py                 # Configurations
│   ├── model.py                  # Model definition
│   └── utils.py                  # Utility functions
├── scripts/                      # Scripts
│   ├── train.py                  # Training script
│   ├── predict.py                # Prediction script
│   ├── api.py                    # API server script
│   ├── api_client.py             # API client for remote predictions
│   ├── mlflow_ui.py              # MLflow UI script
│   ├── build_and_push.py         # Script for building and pushing Docker images to Azure
│   └── deploy_to_azure.py        # Script for deploying to Azure Container Instances
├── tests/                        # Tests
│   ├── test_api.py               # API tests
│   ├── test_model.py             # Model tests
│   ├── test_integration.py       # Integration tests
│   └── test_error_handling.py    # Error handling tests
├── models/                       # Saved models
│   └── model.pth                 # Trained model
├── mlruns/                       # MLflow experiment data
├── data/                         # Data
│   ├── train/                    # Training data
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/                      # Validation data
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/                     # Test data
│       ├── NORMAL/
│       └── PNEUMONIA/
├── docker/                       # Docker files
│   ├── Dockerfile.api            # Dockerfile for API
│   └── Dockerfile.training       # Dockerfile for training
├── .github/                      # GitHub Actions
│   └── workflows/
│       └── test.yml              # CI/CD configuration
├── pyproject.toml                # Poetry dependencies
├── poetry.lock                   # Fixed dependency versions
├── Makefile                      # Automation commands
└── README.md                     # Project documentation
```

## 🚀 Getting Started

### Data Preparation

To work with the project, you need to download the "Chest X-Ray Images (Pneumonia)" dataset from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

After downloading, unzip the archive and move the contents to the `data/` directory in the project root. The data should already be divided into `train/`, `val/` and `test/` folders.

### Installing Dependencies

The project uses Poetry for dependency management:

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

All required dependencies are specified in the `pyproject.toml` file, including:
- torch and torchvision for working with neural networks
- fastapi and uvicorn for API
- scikit-learn for evaluation metrics
- mlflow for experiment tracking
- pytest and other tools for testing

## 📊 Model Training and Evaluation

### Model Architecture

The project uses the **ResNet18** architecture pre-trained on ImageNet with fine-tuning for pneumonia classification:

- Pre-trained ResNet18 model is used as the base
- The last layer is replaced with a new one with 2 outputs (Normal/Pneumonia)
- All layers except the last block (layer4) and fully connected layer are frozen
- Transfer learning is applied for efficient use of a small dataset

### Training Parameters

- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Learning Rate**: 0.001
- **Number of Epochs**: 10
- **Batch Size**: 32
- **Augmentations**: horizontal flip, rotation (±10°), brightness and contrast adjustments

### Training Process

To train the model, run:

```bash
# Using Makefile (with MLflow tracking)
make train-mlflow

# Or standard training
make train

# Or directly with Python
poetry run python scripts/train.py --run_name "my_experiment"
```

Parameters:
- `--data_dir` - path to the data directory (default: ./data)
- `--output_path` - path to save the model (default: ./models/model.pth)
- `--epochs` - number of training epochs (default: 10)
- `--learning_rate` - learning rate (default: 0.001)
- `--run_name` - run name for MLflow (default: auto-generated)
- `--patience` - early stopping patience (default: 5)

### Experiment Tracking with MLflow

The project is integrated with MLflow for experiment tracking and result visualization. MLflow allows:

- Tracking training parameters
- Visualizing metrics in real-time
- Comparing different experiments
- Saving and loading models

To start the MLflow UI, run:

```bash
# Using Makefile
make mlflow-ui

# Or directly
poetry run python scripts/mlflow_ui.py
```

The MLflow interface will be available at: http://127.0.0.1:5000

### Metrics and Evaluation

The model is evaluated using the following metrics:
- **Accuracy**: overall classification accuracy
- **Precision**: precision (proportion of correctly identified positive cases)
- **Recall**: recall (proportion of detected positive cases)
- **F1-score**: harmonic mean between precision and recall

Metrics are monitored on the validation set, and the best model is saved based on minimizing validation loss. Early stopping is implemented to prevent overfitting.

## 🔍 Using the Trained Model

### Inference on Individual Images

For prediction on a single image:

```bash
poetry run python scripts/predict.py --image_path path/to/image.jpg
```

### Starting the API

To start the API server, run:

```bash
# Using Poetry
poetry run python scripts/api.py

# Or using Makefile
make api
```

After starting, the API will be available at `http://localhost:8000/`.

### API Documentation

After starting the server, OpenAPI documentation is available at:
- http://localhost:8000/docs

#### API Endpoints

- `GET /` - API information
- `GET /health` - API health check
- `POST /predict` - image classification

### API Request Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

Example response:
```json
{
  "class_name": "Pneumonia",
  "class_id": 1,
  "probability": 0.9724,
  "filename": "image.jpg",
  "content_type": "image/jpeg"
}
```

## 🌐 Azure Deployment

The project includes scripts for deploying the model to Azure Container Instances (ACI), making the API accessible via the internet.

### Prerequisites

1. Azure CLI installed and configured
2. An active Azure subscription

### Building and Pushing the Docker Image to Azure

```bash
python scripts/build_and_push.py \
  --subscription-id "your-subscription-id" \
  --registry-name "your-registry-name" \
  --image-name "pneumonia-api:v1" \
  --location "westeurope"
```

### Deploying to Azure Container Instances

After pushing the image, deploy it to Azure Container Instances:

```bash
python scripts/deploy_to_azure.py \
  --subscription-id "your-subscription-id" \
  --registry-name "your-registry-name" \
  --image-name "pneumonia-api:v1" \
  --registry-username "username" \
  --registry-password "password" \
  --location "westeurope"
```

## 🐳 Docker

### Building and Running API in Docker

```bash
# Build and run using Makefile
make docker-api

# Or manually
docker build -f docker/Dockerfile.api -t pneumonia-classifier-api:latest .
docker run -p 8000:8000 pneumonia-classifier-api:latest
```

### Training the Model in Docker

```bash
# Build and run using Makefile
make docker-train

# Or manually
docker build -f docker/Dockerfile.training -t pneumonia-classifier-training:latest .
docker run -v $(PWD)/models:/app/models pneumonia-classifier-training:latest
```

## 🧪 Testing

To run tests:

```bash
# Using Poetry
poetry run pytest

# Or using Makefile
make test
```

## 🧹 Linting and Formatting

```bash
# Linting
make lint

# Formatting
make format
```

## 💻 Optimization for Mac with Apple Silicon

The project supports acceleration on Apple Silicon chips (M1/M2/M3) through the MPS backend. For optimal performance and stability on macOS.

## 🔧 Technologies

- **PyTorch**: framework for model training
- **FastAPI**: web framework for API
- **Torchvision**: library for working with images
- **Poetry**: dependency management
- **Docker**: application containerization
- **MLflow**: experiment tracking
- **PyTest**: testing
- **GitHub Actions**: CI/CD

## 🔮 Future Improvements

1. **Hyperparameter Search**: Add GridSearch or Optuna for automatic search of optimal parameters
2. **Advanced Augmentations**: Increase variety of augmentations for model robustness
3. **Other Architectures**: Compare performance with other CNN architectures (EfficientNet, DenseNet)
4. **Model Explainability**: Add Grad-CAM for model decision visualization
5. **Data Drift Monitoring**: Add monitoring for production model 