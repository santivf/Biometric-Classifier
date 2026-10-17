import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_dataset(n_users=10, samples_per_user=20, features=64, random_state=42):
    rng = np.random.default_rng(random_state)
    X, y = [], []
    for i in range(n_users):
        base = rng.random(features)
        for _ in range(samples_per_user):
            sample = base + rng.normal(0, 0.05, features)
            X.append(sample)
            y.append(i)
    return np.array(X), np.array(y)

def train_and_evaluate():
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc, model

