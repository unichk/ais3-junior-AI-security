import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

BASE = os.path.dirname(os.path.abspath(__file__))

def load_split(path):
    arr = np.load(path)
    half = arr.shape[0] // 2
    X = arr.astype(np.float32)
    y = np.array([0] * half + [1] * half)
    return X, y

X_train, y_train = load_split(os.path.join(BASE, "train.npy"))
X_valid, y_valid = load_split(os.path.join(BASE, "valid.npy"))

def train(X, y):
    save = X is None and y is None
    if save:
        X, y = X_train, y_train
    else:
        X = np.concatenate((X, X_train), axis=0)
        y = np.concatenate((y, y_train), axis=0)
    clf = SVC(kernel="linear", C=1.0, max_iter=600000, random_state=0)
    clf.fit(X, y)

    train_acc = accuracy_score(y, clf.predict(X))
    y_pred = clf.predict(X_valid)
    valid_acc = accuracy_score(y_valid, y_pred)
    print(f"\ntrain accuracy: {train_acc:.4f}")
    print(f"val   accuracy: {valid_acc:.4f}\n")
    print(classification_report(y_valid, y_pred, target_names=["fan", "van"], digits=4))
    
    if save:
        import joblib
        joblib.dump({"model": clf}, os.path.join(BASE, "model.joblib"))

if __name__ == "__main__":
    train(None, None)
