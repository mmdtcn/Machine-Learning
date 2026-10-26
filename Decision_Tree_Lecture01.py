import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# Load Dataset (Iris)
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("Features: ", feature_names)
print("Classes: ", target_names)

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42, stratify=y)
# Train the decision tree
clf = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluation Step
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred,
                                                           target_names=target_names))
# Visualize Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=feature_names, class_names=target_names,
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Visualization (Iris Dataset)")
plt.show()

# Feature Importance
importance = clf.feature_importances_
feature_importance = pd.DataFrame({
    'Feature':feature_names,
    'Importance': importance,
    }).sort_values(by='Importance', ascending=False)

print("\nFeature Importance\n")
print(feature_importance)














