<h1 align="center">Deep Learning Project for Volleyball Activity Recognition</h1>


<p align="center">
  <img src="https://i.imgur.com/rhQRxLb.png" alt="Volleyball Activities"  width="80%">
  <img src="https://i.imgur.com/CnDLcFK.jpg" alt="Volleyball Activities"  width="65%">
</p>

 ##  Implemented Paper

| Paper        | Year | Original Paper | Original Implementation | Key Points                        |
|--------------|------|----------------|----------------|-----------------------------------|
| **CVPR 16**| 2016 | [Paper](https://arxiv.org/pdf/1607.02643) | [Implementation](https://github.com/mostafa-saad/deep-activity-rec/tree/master) | Two-stage hierarchical LSTM for group activity recognition      |

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
  - [Ablation Study](#Ablation-Study)
  - [Key Components](#key-components)
- [Dataset](#dataset)
  - [Dataset Labels](#dataset-labels)
  - [Dataset Splits](#dataset-splits)
  - [Dataset Sample](#dataset-sample)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Project Configuration](#project-configuration)
  - [Project Structure](#project-structure)
- [Usage](#usage)
  - [Feature Extraction](#feature-extraction)
  - [Training Models](#training-models)
  - [Available Results](#available-results)
- [Development](#development)
  - [Development Workflow](#development-workflow)
  - [Customization](#customization)
  - [Adding New Models](#adding-new-models)
- [Research Contributions](#research-contributions)
- [References](#references)
- [License](#license)




##  Overview

This project focuses on recognizing complex group activities in volleyball games by analyzing the temporal dynamics and spatial relationships between multiple players. The system can identify various volleyball-specific group activities such as right/left team spikes, sets, passes, and winning points.

##  Problem Statement

Group Activity Recognition in sports videos is challenging because it requires:
- Understanding individual player actions
- Modeling temporal dependencies across frames
- Capturing spatial relationships between multiple players
- Recognizing coordinated team activities

##  Architecture

<img src="https://github.com/mostafa-saad/deep-activity-rec/blob/master/img/fig1.png" alt="Figure 1" height="400" >

**Figure 1**: High level figure for group activity recognition via a hierarchical model. Each person in a scene is modeled using a temporal model that captures his/her dynamics, these models are integrated into a higher-level model that captures scene-level activity.

<img src="https://github.com/mostafa-saad/deep-activity-rec/blob/master/img/fig2-b.png" alt="Figure 2" height="400" >

**Figure 2**: Detailed figure for the model. Given tracklets of K-players, we feed each tracklet in a CNN, followed by a person LSTM layer to represent each player's action. We then pool over all people's temporal features in the scene. The output of the pooling layer is feed to the second LSTM network to identify the whole teams activity.

<img src="https://github.com/mostafa-saad/deep-activity-rec/blob/master/img/fig3.jpg" alt="Figure 3" height="400" >

**Figure 3**: Previous basic mode drops spatial information. In updated model, 2-group pooling to capture spatial arrangements of players.

### Ablation Study

| Model | Description | Architecture | Test Accuracy |
|-------|-------------|--------------|---------------|
| **Baseline 1** | Simple ResNet-50 | Single CNN for frame-level classification | **74.83%** |
| **Baseline 3A** | Feature extraction model | ResNet-based feature extractor for individual actions | **78.27%** |
| **Baseline 3B** | Enhanced feature model | Improved version of 3A with better feature representation | **82.12%** |
| **Baseline 4** | Temporal modeling | LSTM-based temporal sequence modeling | **81.08%** |
| **Baseline 5** | Multi-stream approach | Multiple input streams for different modalities | **83.40%** |
| **Baseline 6** | Attention mechanism | Attention-based temporal modeling | **77.86%** |
| **Baseline 7** | Hierarchical modeling | Multi-level temporal and spatial modeling | **86.46%** |
| **Baseline 8** | Advanced LSTM | Dual LSTM with team-based aggregation | **89.08%** |

### Key Components

- **Feature Extraction**: Uses pre-trained ResNet models to extract 2048-dimensional features from player bounding boxes
- **Temporal Modeling**: LSTM networks to capture sequential dependencies
- **Spatial Aggregation**: Team-based feature aggregation for group activity recognition
- **Multi-level Classification**: Hierarchical approach from individual actions to group activities

## Dataset
We used a volleyball dataset introduced in the aforementioned paper. The dataset consists of:
- **Videos**: 55 YouTube volleyball videos.
- **Frames**: 4830 annotated frames, each with bounding boxes around players and labels for both individual actions and group activities.

### Dataset Labels

<table>
  <tr>
    <!-- We ensure each cell is top-aligned -->
    <td valign="top">

#### Group Activity Classes

| Class          | Instances |
|----------------|-----------|
| Right set      | 644       |
| Right spike    | 623       |
| Right pass     | 801       |
| Right winpoint | 295       |
| Left winpoint  | 367       |
| Left pass      | 826       |
| Left spike     | 642       |
| Left set       | 633       |

</td>
    <td valign="top">

#### Action Classes

| Class    | Instances |
|----------|-----------|
| Waiting  | 3601      |
| Setting  | 1332      |
| Digging  | 2333      |
| Falling  | 1241      |
| Spiking  | 1216      |
| Blocking | 2458      |
| Jumping  | 341       |
| Moving   | 5121      |
| Standing | 38696     |

</td>
  </tr>
</table>



### Dataset Splits
  - Training Set: 2/3 of the videos.
    - Train Videos: 1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54. 
  - Validation Set: 15 videos.
    - Validation Videos: 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51.
  - Test Set: 1/3 of the videos.
    - Test Videos: 4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47.

### Dataset Sample
<p align="center">

<img  src="https://i.imgur.com/DUhaofS.gif" alt="B8" width="75%">
</p>

The dataset is available for download at [GitHub Deep Activity Rec](https://github.com/mostafa-saad/deep-activity-rec#dataset), or on Kaggle [here](https://www.kaggle.com/datasets/ahmedmohamed365/volleyball/data?select=volleyball_)

##  Getting Started

### Prerequisites

- Python 3.13.7+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- CUDA (recommended for training)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd volleyball
```

2. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install dependencies using uv:
```bash
uv sync
```

This will automatically:
- Create a virtual environment
- Install all dependencies from `pyproject.toml`
- Set up the project for development

### Project Configuration

The project uses modern Python packaging with `pyproject.toml`:

- **Dependencies**: PyTorch, torchvision, Pillow, matplotlib, scikit-learn
- **Python Version**: 3.13.7+
- **Code Quality**: Ruff for linting and formatting
- **Package Manager**: uv for fast dependency resolution

### Project Structure

```
volleyball/
├── models/                 # Model architectures (8 baselines)
├── datasets/              # Dataset loaders and utilities
├── trainers/              # Training scripts for each baseline
├── utils/                 # Utility functions and helpers
├── extract_features.py    # Feature extraction pipeline
├── constants.py           # Configuration constants
├── trained_models/        # Pre-trained model weights
├── confusion_matrix/      # Confusion matrix visualizations
├── loss_accuracy/         # Training curves
└── logs/                  # Training logs
```

##  Usage

### 1. Feature Extraction

Extract deep features from video frames using pre-trained models:

```bash
uv run python extract_features.py
```

This script:
- Loads video frames and player bounding boxes
- Crops individual player regions
- Extracts 2048-dimensional features using Baseline 3A model
- Saves features for training/validation/test splits

### 2. Training Models

Train any of the 8 baseline models:

```bash
# Train Baseline 1 (ResNet-50)
uv run python trainers/train_b1.py

# Train Baseline 8 (Advanced LSTM)
uv run python trainers/train_b8.py
```


### Available Results

- **Training Curves**: Loss and accuracy plots for all baselines
- **Confusion Matrices**: Detailed classification performance analysis
- **Model Weights**: Pre-trained models for immediate use
- **Logs**: Detailed training logs for reproducibility

##  Research Contributions

This implementation provides:

1. **Comprehensive Baselines**: 8 different approaches to GAR
2. **Volleyball-Specific Modeling**: Domain-adapted for sports analysis
3. **Temporal Dynamics**: Advanced LSTM-based sequence modeling
4. **Spatial Relationships**: Team-based feature aggregation
5. **Reproducible Results**: Complete training and evaluation pipeline

### Development

### Development Workflow

```bash
# Activate the virtual environment
uv shell

# Run scripts in the virtual environment
uv run python script.py

# Add new dependencies
uv add package-name

# Update dependencies
uv sync

# Run linting and formatting
uv run ruff check .
uv run ruff format .
```

### Customization

### Adding New Models

1. Create a new model class in `models/`
2. Implement the forward pass
3. Create a training script in `trainers/`
4. Update constants if needed


##  References

- [Original Paper](https://www.cs.sfu.ca/~mori/research/papers/ibrahim-cvpr16.pdf)
- [CVPR 2016](https://cvpr2016.thecvf.com/)
- [Group Activity Recognition Survey](https://arxiv.org/abs/2006.06966)


##  License

This project is for research purposes. Please cite the original paper if you use this implementation in your research.
