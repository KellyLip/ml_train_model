# Random Forest Model Training and Validation

This repository contains scripts to train and validate a Random Forest model with hyperparameter optimization using `RandomizedSearchCV`. The model is trained on a designated training dataset, validated on a separate validation dataset, and the final trained model is saved for later use.

---

## Repository Contents

- **`train_model_RFR.py`**: A Python script that:
  - Loads a dataset from a CSV file.
  - Splits the dataset into training and validation sets.
  - Optimizes Random Forest hyperparameters using `RandomizedSearchCV`.
  - Trains the model with the best hyperparameters on the training set.
  - Validates the model on the validation set and reports its performance.
  - Saves the final trained model to a specified file.
  
- **`run_train_model_RFR.sh`**: A Bash script to execute the Python script with specified parameters, making it easier to reuse and automate.
  
- **`requirements.txt`**: A file listing the Python libraries required to run the script.

---

## Features and Workflow

1. **Dataset Loading**:  
   The dataset must be in CSV format. You provide the file path, feature column names, and the target column as input parameters.

2. **Train-Validation Split**:  
   The dataset is split into:
   - **Training Set**: Used to train the model.
   - **Validation Set**: Used to evaluate the model's performance.

3. **Hyperparameter Optimization**:  
   The script uses `RandomizedSearchCV` to optimize key hyperparameters of the Random Forest model:
   - `n_estimators`: Number of trees in the forest.
   - `max_depth`: Maximum depth of the trees.
   - `min_samples_split`: Minimum number of samples required to split a node.
   - `min_samples_leaf`: Minimum number of samples required in a leaf node.

   The best hyperparameters are selected based on the negative Mean Absolute Error (MAE) score.

4. **Model Training**:  
   The final model is trained on the training set using the best hyperparameters from the optimization step.

5. **Model Validation**:  
   The trained model is validated on the validation set. The script calculates the Mean Absolute Error (MAE) to measure performance.

6. **Model Saving**:  
   The trained model is saved as a `.pkl` file using `joblib`, allowing you to reuse it without retraining.

---

## Requirements

- **Python**: Version 3.6 or later.
- **Python Libraries**:
  - `pandas`: For data manipulation and loading.
  - `scikit-learn`: For machine learning models, hyperparameter optimization, and evaluation.
  - `joblib`: For saving and loading the trained model.

Install the required libraries using:
```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Prepare Your Dataset
- Your dataset must be in CSV format.
- Ensure the file includes the columns for the features and the target variable.

### Step 2: Modify Parameters in `run_train_model.sh`
Open the `run_train_model.sh` script and update the following variables:
- `DATA_PATH`: Path to your dataset file (e.g., `../path/to/data.csv`).
- `FEATURES`: List of feature column names in Python list format (e.g., `['feature1', 'feature2']`).
- `TARGET`: Name of the target column (e.g., `target_column`).
- `OUTPUT_MODEL`: File name to save the trained model (e.g., `trained_random_forest.pkl`).

Example:
```bash
DATA_PATH="../path/to/data.csv"
FEATURES="['feature1', 'feature2', 'feature3']"
TARGET="target_column"
OUTPUT_MODEL="trained_random_forest.pkl"
```

### Step 3: Run the Training Script
Run the bash script:
```bash
bash run_train_model.sh
```

This command will:
- Load the dataset.
- Train and validate the model.
- Save the final trained model.

### Step 4: View Output
The script prints:
1. **Best Hyperparameters**: The optimal values found using `RandomizedSearchCV`.
2. **Validation MAE**: The Mean Absolute Error on the validation set.
3. **Model Save Confirmation**: The path where the trained model is saved.

Example Output:
```plaintext
Optimizing hyperparameters...
Fitting 3 folds for each of 10 candidates, totalling 30 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
...
Best parameters: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
Best score (negative MAE): -1234.56

Training and validating the model...
Validation MAE: 1123.45
Model saved to trained_random_forest.pkl
```

### Step 5: Reuse the Model
You can load the saved model for predictions using `joblib`:
```python
import joblib

# Load the model
model = joblib.load("trained_random_forest.pkl")

# Use the model for predictions
predictions = model.predict(new_data)
```

---

## Additional Notes

### Model Evaluation
- The **Mean Absolute Error (MAE)** is used to measure the model's performance. Lower MAE values indicate better predictions.
- If required, you can modify the script to include additional evaluation metrics like RMSE or R^2.

### Hyperparameter Optimization
- The number of parameter combinations tested is controlled by `n_iter` in `RandomizedSearchCV`.
- You can increase `n_iter` for a more exhaustive search but note that this will increase runtime.
