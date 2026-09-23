import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve
from pathlib import Path
from sklearn.metrics import roc_auc_score


IMAGES_PATH = Path() / "images" / "classification"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Load MNIST dataset - add as_frame=False to get numpy arrays
mnist = fetch_openml(name='mnist_784', version=1, as_frame=False)

print(mnist.DESCR)
mnist.keys()

X, y = mnist.data, mnist.target
print(X)
print(X.shape)
print(y)
print(y.shape)

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

some_digit = X[0]
plot_digit(some_digit)
save_fig("some_digit_plot")
plt.show()
print(y[0])


# extra code – this cell generates and saves as Figure
plt.figure(figsize=(9, 9))
for idx, image_data in enumerate(X[:100]):
    plt.subplot(10, 10, idx + 1)
    plot_digit(image_data)
plt.subplots_adjust(wspace=0, hspace=0)
save_fig("more_digits_plot", tight_layout=False)
plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Training a Binary Classifier
y_train_5 = (y_train == '5')  # True for all 5s, False for all other digits
y_test_5 = (y_test == '5')
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([some_digit]))

'''
# Performance Measures
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# Confusion Matrix
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)

y_train_perfect_predictions = y_train_5  # pretend we reached perfection
confusion_matrix(y_train_5, y_train_perfect_predictions)

# Precision & Recall
precision_score(y_train_5, y_train_pred)
# extra code – this cell also computes the precision: TP / (FP + TP)
print(cm[1, 1] / (cm[0, 1] + cm[1, 1]))

recall_score(y_train_5, y_train_pred)

# extra code – this cell also computes the recall: TP / (FN + TP)
print(cm[1, 1] / (cm[1, 0] + cm[1, 1]))

# Precision/Recall Trade-off
y_scores = sgd_clf.decision_function([some_digit])

threshold = 0
y_some_digit_pred = (y_scores > threshold)

threshold = 3000
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
# The ROC Curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")

plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.grid()
plt.axis([0, 1, 0, 1])
plt.legend(loc="lower right", fontsize=13)
save_fig("roc_curve_plot")
plt.show()

print(roc_auc_score(y_train_5, y_scores))
'''