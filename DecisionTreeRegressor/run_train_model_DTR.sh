#!/bin/bash

# User-defined values
DATA_PATH="path/to/your/dataset.csv"  # Replace with your dataset path
TARGET="target_column_name"          # Replace with your target column name
FEATURES="col1,col2,col3"            # Replace with comma-separated feature column names
MODEL_NAME="best_decision_tree_model.pkl"  # Replace with desired model file name

# Run the Python script with the specified parameters
python train_decision_tree.py "$DATA_PATH" "$TARGET" "$FEATURES" "$MODEL_NAME"
