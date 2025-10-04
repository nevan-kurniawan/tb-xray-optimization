# Optimizing Custom CNN Blocks for TB X-ray Analysis via Bayesian Optimization

This repository contains the official source code and experimental artifacts for the paper:

> **Optimizing Custom Convolutional Blocks for Pre-trained CNNs in Tuberculosis X-ray Analysis via Bayesian Optimization**  
> **Authors**: _Nicholas Nevan Kurniawan, Reynard Amadeus Joshua, Ivan Sebastian Edbert, Alvina Aulia_  
> **Conference:** CENIM 2025

## Abstract

This study optimizes Convolutional Neural Network (CNN) architectures for tuberculosis (TB) X-ray detection. Leveraging transfer learning, ResNet50, Inception V3, and DenseNet121 models, which were pre-trained on RadImageNet, were fine-tuned with appended custom convolutional blocks. Their hyperparameters (filter count, kernel size, pooling type) were meticulously tuned using Bayesian Optimization, targeting the F1-score. Evaluated on a public Kaggle TB dataset, the individually optimized models all achieved excellent performance, with key metrics consistently exceeding 0.90. These results validate Bayesian Optimization for tailoring custom layers, producing robust, high-performing TB detection models and highlighting the need for architecture-specific tuning.

## Repository Structure

The project is organized to clearly separate inputs, code, and outputs.

```
tb-xray-optimization/
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   └── tuberculosis-tb-chest-xray-dataset
│       └──TB_Chest_Radiography_Database/
│           ├── Normal/
│           └── Tuberculosis/
│
├── models/
│   ├── resnet50_final_model.weights.h5
│   ├── inceptionv3_final_model.weights.h5
│   └── densenet121_final_model.weights.h5
│
├── notebooks/
│   ├── 1_bayesian_optimization/
│   │   ├── bo-densenet121.ipynb
│   │   ├── bo-inceptionv3.ipynb
│   │   └── bo-resnet50.ipynb
│   └── 2_final_model_training/
│       ├── densenet121-cnn-tb.ipynb
│       ├── inceptionv3-cnn-tb.ipynb
│       └── resnet50-cnn-tb.ipynb
│
├── results/
│   ├── optuna_study_tb_ResNet50_radimagenet.db
│   ├── best_params_tb_ResNet50_radimagenet.json
│   └── (and other .db and .json files)
│
└── weights/
    ├── RadImageNet-ResNet50_notop.h5
    ├── RadImageNet-InceptionV3_notop.h5
    └── RadImageNet-DenseNet121_notop.h5
```

-   **`data/`**: Stores the input dataset.
-   **`weights/`**: Stores the pre-trained RadImageNet weights.
-   **`notebooks/`**: Contains all experimental code, organized into two sequential phases.
-   **`results/`**: Contains all generated outputs, including Optuna databases (`.db`) and best hyperparameter files (`.json`).
-   **`models/`**: Stores the final, trained `.h5` model files.

## Setup and Installation

To set up the environment and run the experiments, please follow these steps.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR-USERNAME]/tb-xray-optimization.git
    cd tb-xray-optimization
    ```

2.  **Download the Data**
    -   Download the "Tuberculosis (TB) Chest X-ray Database" from [this Kaggle page](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset).
    -   Extract the archive and place the `TB_Chest_Radiography_Database` folder inside the `data/tuberculosis-tb-chest-xray-dataset` directory.

3.  **Create a Virtual Environment and Install Dependencies**
    It is highly recommended to use a virtual environment.
    ```bash
    # Create a virtual environment (e.g., using venv)
    python -m venv .venv
    # Activate it
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

    # Install all required packages
    pip install -r requirements.txt
    ```

## Experimental Workflow

The experiment is divided into two sequential phases, executed via the notebooks in the `notebooks/` directory.

### Phase 1: Bayesian Optimization

The three notebooks located in `notebooks/1_bayesian_optimization/` are used to find the optimal hyperparameters for the custom convolutional blocks for each of the three base models (ResNet50, InceptionV3, DenseNet121).

-   Running `bo-resnet50.ipynb` will execute a 50-trial Optuna study.
-   The full study history is saved to a `.db` file in the `results/` directory.
-   The best hyperparameter set is saved to a `.json` file in the `results/` directory.

### Phase 2: Final Model Training

After running the optimization, the three notebooks in `notebooks/2_final_model_training/` are used to train the final models.

-   Each notebook (e.g., `resnet50-cnn-tb.ipynb`) loads the corresponding `best_params.json` file from the `results/` directory.
-   It then builds the model with these optimal hyperparameters, trains it, and evaluates it on the hold-out test set.
-   The final trained model is saved as an `.weights.h5` file in the `models/` directory.

## A Note on Reproducibility

The initial Bayesian Optimization experiments were conducted without a fixed random seed. Therefore, the hyperparameter search itself is not bit-for-bit reproducible.

However, the final model training and evaluation process **is fully reproducible**. Using the optimal hyperparameters provided in the `results/` directory and the random seed (`SEED = 42`) set in the final training notebooks, anyone can replicate the exact training process and achieve the performance metrics reported in the paper.