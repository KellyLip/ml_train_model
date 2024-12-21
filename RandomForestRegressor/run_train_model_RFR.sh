#!/bin/bash

# Variables
DATA_PATH="../path/to/data.csv"
FEATURES="['column1', 'column2', 'column3', 'column4', 'column5', 'column6', 'column7']"
TARGET="column_to_predict"
OUTPUT_MODEL="trained_random_forest.pkl"

# Run the Python script
python train_model.py --data_path "$DATA_PATH" --features "$FEATURES" --target "$TARGET" --output_model "$OUTPUT_MODEL"
