import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sbn

df = pd.read_csv('classification/Iris.csv')

df["Species"] = df["Species"].replace({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
})

X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=1
)

print(f'[INFO] Train Shape :: {train_x.shape}')
print(f'[INFO] Test Shape :: {test_x.shape}')

class MultiClassLogisticRegression:
    def __init__(self, lr=0.01, epochs=5000):
        self.lr = lr
        self.epochs = epochs
        self.betas = None

    def concat_ones(self, x):
        return np.c_[np.ones((x.shape[0], 1)), x]

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def to_categorical(self, y):
        y = y.astype(int)
        return np.eye(len(np.unique(y)))[y]

    def fit(self, x, y):
        x = self.concat_ones(x)
        y = self.to_categorical(y)

        n_samples, n_features = x.shape
        n_classes = y.shape[1]

        self.betas = np.zeros((n_features, n_classes))

        for _ in range(self.epochs):
            scores = x @ self.betas
            probs = self.softmax(scores)
            gradient = (x.T @ (probs - y)) / n_samples
            self.betas -= self.lr * gradient

    def predict_proba(self, x):
        x = self.concat_ones(x)
        return self.softmax(x @ self.betas)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

mlr = MultiClassLogisticRegression(lr=0.01, epochs=5000)
mlr.fit(train_x, train_y)

train_pred = mlr.predict(train_x)
test_pred = mlr.predict(test_x)

train_cm = confusion_matrix(train_y, train_pred)

plt.figure(figsize=(5, 5))
sbn.heatmap(
    train_cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    xticklabels=['Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica'],
    yticklabels=['Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Train Confusion Matrix')
plt.show()

train_accuracy = np.trace(train_cm) / np.sum(train_cm)
print(f"Train Accuracy: {train_accuracy:.4f}")

test_cm = confusion_matrix(test_y, test_pred)

plt.figure(figsize=(5, 5))
sbn.heatmap(
    test_cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    cbar=False,
    xticklabels=['Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica'],
    yticklabels=['Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Test Confusion Matrix')
plt.show()

test_accuracy = np.trace(test_cm) / np.sum(test_cm)
print(f"Test Accuracy: {test_accuracy:.4f}")
