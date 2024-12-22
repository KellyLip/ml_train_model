import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import sys

def main(data_path, target, features, model_name):
    # Load dataset
    data = pd.read_csv(data_path)

    # Define target and features
    X = data[features]
    y = data[target]

    # Split data into training and validation sets
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # Define parameter grid for GridSearchCV
    param_grid = {
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform GridSearchCV to find the best parameters
    grid_search = GridSearchCV(estimator=DecisionTreeRegressor(random_state=1), param_grid=param_grid, cv=3)
    grid_search.fit(train_X, train_y)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate the model on validation data
    val_predictions = best_model.predict(val_X)
    val_mae = mean_absolute_error(val_y, val_predictions)

    # Save the model as a pickle file
    with open(model_name, 'wb') as file:
        pickle.dump(best_model, file)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Validation MAE: {val_mae:.2f}")
    print(f"Model saved as {model_name}")

if __name__ == "__main__":
    # Read arguments from the command line
    if len(sys.argv) < 5:
        print("Usage: python train_decision_tree.py <data_path> <target_column> <features> <model_name>")
        print("Example: python train_decision_tree.py data.csv target 'col1,col2,col3' best_model.pkl")
        sys.exit(1)

    data_path = sys.argv[1]
    target = sys.argv[2]
    features = sys.argv[3].split(',')
    model_name = sys.argv[4]

    main(data_path, target, features, model_name)
