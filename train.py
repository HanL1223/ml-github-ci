# train.py
def train_model():
    """Train a simple RandomForest model on Iris dataset and save it."""
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    import os

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train model
    clf = RandomForestClassifier()
    clf.fit(X, y)

    # Save model
    model_path = os.path.join(os.getcwd(), "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"Model trained and saved at {model_path}")


if __name__ == "__main__":
    train_model()
