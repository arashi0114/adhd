# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ADHD cognitive impairment prediction project using PyTorch deep learning. It supports both classification and regression tasks for predicting cognitive impairments from ELSA (English Longitudinal Study of Ageing) data.

## Development Commands

### Running the Project
```bash
python main.py
```

### Environment Setup
```bash
pip install -r requirements.txt
```

### Data Processing
The project expects STATA files at `data/raw/h_elsa_g3.dta`. Jupyter notebooks for data exploration are in the `notebook/` directory.

## Architecture

### Core Pipeline (main.py)
1. **Data Loading** (`src.data.load`) - Loads and filters ELSA data with minimum wave and sparsity requirements
2. **Preprocessing** (`src.data.preprocess`) - Handles missing values, standardization, and outliers 
3. **Dataset Creation** (`src.data.dataset`) - Creates PyTorch datasets with train/val/test splits
4. **Model Training** (`src.training.train`) - Trains with early stopping and learning rate scheduling
5. **Evaluation** (`src.training.evaluate`) - Comprehensive evaluation with metrics and visualizations

### Configuration System (`src/config.py`)
- YAML-based configuration in `config/default.yaml`
- Automatic device detection (CUDA/CPU)
- Auto-creation of output directories
- Task-specific output dimension setting
- Configuration validation and nested updates

### Models (`src/models/`)
- **Classification** (`classification.py`) - Multi-layer neural network for binary cognitive impairment classification
- **Regression** (`regression.py`) - Multi-layer neural network for cognitive score prediction
- Both support configurable hidden layers, dropout, and activation functions

### Key Configuration Parameters
- `model.task_type`: "classification" or "regression"
- `data.filters.min_waves`: Minimum number of waves required (default: 8)
- `data.filters.min_non_sparsity`: Minimum data density required (default: 0.7)
- `train.device`: "cuda" or "cpu" (auto-detected)
- `train.early_stopping.patience`: Early stopping patience (default: 15)

### Output Structure
- `outputs/models/` - Saved model checkpoints
- `outputs/results/` - Evaluation results and visualizations
- `outputs/logs/` - Training logs

## Development Notes

- The project uses Chinese comments and documentation
- No existing test framework - check main.py execution for validation
- Models automatically save best checkpoints during training
- Configuration validation ensures required parameters are present
- Device selection automatically falls back to CPU if CUDA unavailable