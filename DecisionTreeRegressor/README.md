# Decision Tree Regressor Training Repository

This repository provides a simple and efficient workflow for training a Decision Tree Regressor model using hyperparameter tuning with `GridSearchCV`. The repository is designed to save the trained model as a pickle file for future use.

## Repository Contents

- **`requirements.txt`**: Lists the Python dependencies required for the project.
- **`train_model_DTR.py`**: A Python script for training the model. It uses command-line arguments to accept dataset path, target column, feature columns, and the output file name for the trained model.
- **`run_train_model_DTR.sh`**: A customizable bash script to run the Python script. Users can modify this script to specify their own dataset, target, features, and model name.

## Features and Workflow

1. **GridSearchCV for Hyperparameter Tuning**:
   - Automatically finds the best parameters for the Decision Tree Regressor by testing multiple configurations.
2. **Customizable Workflow**:
   - Modify the bash script (`run_train_model_DTR.sh`) to define your dataset path, target column, features, and the name of the saved model.
3. **Save Trained Model**:
   - Saves the trained model as a pickle file for easy reuse.
4. **Validation Metrics**:
   - Displays the best hyperparameters and validation mean absolute error (MAE) after training.

### Workflow
1. Users modify the bash script to define input parameters.
2. Run the bash script to execute the Python training script.
3. The Python script trains the model, tunes hyperparameters, and saves the trained model.

## Requirements

- Python 3.7+
- Required Python libraries are listed in `requirements.txt`:
  ```plaintext
  pandas
  scikit-learn
  ```

Install the dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Clone the Repository
Clone this repository to your local machine:
```bash
git clone <repository-url>
cd <repository-folder>
```

### Step 2: Install Requirements
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Step 3: Customize the Bash Script
Open `run_train_model_DTR.sh` and modify the following variables with your own values:
- `DATA_PATH`: Path to your dataset file.
- `TARGET`: Target column to predict.
- `FEATURES`: Comma-separated list of feature columns.
- `MODEL_NAME`: Name of the file to save the trained model.

Example:
```bash
DATA_PATH="data/my_dataset.csv"
TARGET="price"
FEATURES="feature1,feature2,feature3"
MODEL_NAME="trained_model.pkl"
```

### Step 4: Run the Script
Run the bash script to train the model:
```bash
./run_train_model_DTR.sh
```

### Step 5: Use the Trained Model
The trained model will be saved as a `.pkl` file in the repository folder. Load it in Python to make predictions:
```python
import pickle

with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# Example: Make predictions
predictions = model.predict(new_data)
```

## Additional Notes

- **Supported Formats**: Ensure your dataset is in CSV format.
- **Error Handling**: The Python script provides error messages for missing or invalid arguments.
- **Customization**: The Python script can also be called directly for custom workflows:
  ```bash
  python train_model_DTR.py <data_path> <target_column> <features> <model_name>
  ```
