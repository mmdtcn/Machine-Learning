# Multiclass Softmax Regression on MNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load MNIST (digits 0–9, 8x8 images from sklearn for speed)
digits = load_digits()
X, y = digits.data, digits.target

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))

X_train, X_test, y_train, y_test = X[:1000], X[1000:], y[:1000], y[1000:]

# 2. Train multinomial logistic regression (softmax)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# 3. Predictions
y_pred = model.predict(X_test)

# 4. Evaluation
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# 5. Show an example prediction
sample_index = 123
plt.imshow(digits.images[sample_index], cmap='gray')
plt.title(f"True label: {y[sample_index]}")
plt.show()

# Predicted probabilities (softmax outputs)
probs = model.predict_proba([X[sample_index]])[0]
print("Predicted probabilities (softmax):")
for digit_class, prob in enumerate(probs):
    print(f"Digit {digit_class}: {prob:.3f}")

print("\nPredicted label:", np.argmax(probs))
