import pandas as pd
import argparse
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def load_dataset(data_path, features, target):
    """Load the dataset and split into features and target."""
    data = pd.read_csv(data_path)
    X = data[features]
    y = data[target]
    return X, y

def optimize_hyperparameters(train_X, train_y):
    """Perform Randomized Search to optimize Random Forest hyperparameters."""
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=1),
        param_distributions=param_dist,
        n_iter=10,  # Number of parameter combinations to test
        cv=3,  # 3-fold cross-validation
        scoring='neg_mean_absolute_error',
        random_state=1,
        verbose=2
    )
    
    random_search.fit(train_X, train_y)
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best score (negative MAE): {random_search.best_score_:.2f}")
    return random_search.best_params_

def train_and_validate(train_X, train_y, val_X, val_y, best_params):
    """Train the model on training data and validate on validation data."""
    model = RandomForestRegressor(**best_params, random_state=1)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds)
    print(f"Validation MAE: {mae:.2f}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train and optimize a Random Forest model.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset file.")
    parser.add_argument('--features', type=str, required=True, help="Comma-separated list of feature column names.")
    parser.add_argument('--target', type=str, required=True, help="Name of the target column.")
    parser.add_argument('--output_model', type=str, default="final_model.pkl", help="Path to save the trained model.")
    
    args = parser.parse_args()
    features = eval(args.features)  # Convert string to list

    # Load and split the dataset
    X, y = load_dataset(args.data_path, features, args.target)
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Optimize hyperparameters
    print("Optimizing hyperparameters...")
    best_params = optimize_hyperparameters(train_X, train_y)

    # Train and validate the model
    print("\nTraining and validating the model...")
    final_model = train_and_validate(train_X, train_y, val_X, val_y, best_params)
    
    # Save the trained model
    import joblib
    joblib.dump(final_model, args.output_model)
    print(f"Model saved to {args.output_model}")

if __name__ == "__main__":
    main()
