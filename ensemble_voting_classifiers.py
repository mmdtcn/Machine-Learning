# === Ensemble Learning Example: Base Models + Voting Classifiers ===
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------
# 1. Create Dataset
# -----------------------------------------------------------
X, y = make_classification(
    n_samples=500, n_features=5, n_informative=3, n_redundant=0,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------------------------------------
# 2. Define Base Models
# -----------------------------------------------------------
log_clf = LogisticRegression()
tree_clf = DecisionTreeClassifier(random_state=42)
svm_clf = SVC()  # default: no probability output

# Train each model individually
for name, clf in [("Logistic Regression", log_clf),
                  ("Decision Tree", tree_clf),
                  ("SVM", svm_clf)]:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# -----------------------------------------------------------
# 3. Hard Voting Classifier
# -----------------------------------------------------------
voting_hard = VotingClassifier(
    estimators=[('lr', log_clf), ('dt', tree_clf), ('svm', svm_clf)],
    voting='hard'  # majority vote (mode)
)
voting_hard.fit(X_train, y_train)
y_pred_hard = voting_hard.predict(X_test)
print(f"\nHard Voting Accuracy: {accuracy_score(y_test, y_pred_hard):.3f}")

# -----------------------------------------------------------
# 4. Soft Voting Classifier
# -----------------------------------------------------------
# SVC must support probability=True for soft voting
log_clf2 = LogisticRegression()
tree_clf2 = DecisionTreeClassifier(random_state=42)
svm_clf2 = SVC(probability=True)

voting_soft = VotingClassifier(
    estimators=[('lr', log_clf2), ('dt', tree_clf2), ('svm', svm_clf2)],
    voting='soft'  # uses averaged predicted probabilities
)
voting_soft.fit(X_train, y_train)
y_pred_soft = voting_soft.predict(X_test)
print(f"Soft Voting Accuracy: {accuracy_score(y_test, y_pred_soft):.3f}")
