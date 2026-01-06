import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt


# =========================
# 1. LOAD DATASET
# =========================
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target


# =========================
# 2. SPLIT DATA
# =========================
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 3. BUILD MODEL
# =========================
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# 4. EVALUATE MODEL
# =========================
y_pred = model.predict(X_test)

print("===== HASIL EVALUASI =====")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# =========================
# 5. VISUALIZE TREE
# =========================
plt.figure(figsize=(12,8))
tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=data.target_names,
    filled=True
)
plt.show()
