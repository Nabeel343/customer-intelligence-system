import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# -----------------------------
# STEP 1: Create Industry Dataset
# -----------------------------
data = {
    "CustomerID": range(1, 21),
    "MonthlySpend": [500, 800, 200, 1200, 1500, 300, 700, 1600, 400, 1000,
                     1100, 600, 900, 1300, 1400, 350, 750, 1550, 450, 1050],
    "PurchaseFrequency": [2, 5, 1, 6, 7, 1, 4, 8, 2, 5,
                          6, 3, 5, 7, 6, 2, 4, 8, 2, 5],
    "AvgOrderValue": [250, 300, 200, 400, 450, 150, 280, 500, 220, 350,
                      370, 260, 310, 420, 430, 180, 290, 480, 210, 360],
    "TenureMonths": [12, 24, 6, 36, 40, 8, 20, 45, 10, 30,
                     32, 18, 25, 38, 42, 9, 22, 44, 11, 28],
    "DiscountUsage": [0.2, 0.5, 0.1, 0.6, 0.7, 0.2, 0.4, 0.8, 0.3, 0.5,
                      0.6, 0.3, 0.5, 0.7, 0.6, 0.2, 0.4, 0.8, 0.3, 0.5],
    "Returns": [1, 2, 0, 3, 2, 1, 1, 4, 0, 2,
                2, 1, 2, 3, 2, 1, 1, 4, 0, 2],
    "LoyalCustomer": [0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
                      1, 0, 1, 1, 1, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
print("\n1) Customer Dataset:")
print(df)


# ------------------------------------------------
# STEP 2: Unsupervised Learning (Customer Segments)
# ------------------------------------------------
X_cluster = df[["MonthlySpend", "PurchaseFrequency"]]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["CustomerSegment"] = kmeans.fit_predict(X_cluster)

print("\n2) Customer Segmentation using K-Means:")
print(df[["CustomerID", "MonthlySpend", "PurchaseFrequency", "CustomerSegment"]])
# 📊 Customer Segmentation Plot
plt.figure()
plt.scatter(df["MonthlySpend"], df["PurchaseFrequency"], c=df["CustomerSegment"])
plt.xlabel("Monthly Spend")
plt.ylabel("Purchase Frequency")
plt.title("Customer Segmentation (K-Means)")
plt.show()

# -----------------------------------------------
# STEP 3: Supervised Learning (Logistic Regression)
# -----------------------------------------------
X = df[["MonthlySpend", "PurchaseFrequency"]]
y = df["LoyalCustomer"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

log_predictions = log_model.predict(X_test)


# -----------------------------
# STEP 4: Model Evaluation
# -----------------------------
print("\n3) Logistic Regression Performance:")

print("Accuracy :", accuracy_score(y_test, log_predictions))
print("Precision:", precision_score(y_test, log_predictions, zero_division=0))
print("Recall   :", recall_score(y_test, log_predictions, zero_division=0))
print("F1 Score :", f1_score(y_test, log_predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, log_predictions))
# 📊 Confusion Matrix Heatmap
cm = confusion_matrix(y_test, log_predictions)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------------------------------
# STEP 5: Model Fine-Tuning
# ---------------------------------------
log_model_tuned = LogisticRegression(C=0.5)
log_model_tuned.fit(X_train, y_train)

tuned_predictions = log_model_tuned.predict(X_test)

print("\n4) Tuned Logistic Regression Accuracy:")
print("Accuracy:", accuracy_score(y_test, tuned_predictions))


# -----------------------------
# STEP 6: Model Comparison
# -----------------------------

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

# Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_predictions = tree.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)

# SVM
svm = SVC()
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)

# Gradient Boosting
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
gb_predictions = gb.predict(X_test)

print("\n--- Model Comparison ---")
print(f"Logistic Regression: {accuracy_score(y_test, log_predictions):.2f}")
print(f"KNN: {accuracy_score(y_test, knn_predictions):.2f}")
print(f"Decision Tree: {accuracy_score(y_test, tree_predictions):.2f}")
print(f"Random Forest: {accuracy_score(y_test, rf_predictions):.2f}")
print(f"SVM: {accuracy_score(y_test, svm_predictions):.2f}")
print(f"Gradient Boosting: {accuracy_score(y_test, gb_predictions):.2f}")
# 📊 Model Comparison Graph
models = ["Logistic", "KNN", "Decision Tree", "Random Forest", "SVM", "Gradient Boosting"]
scores = [
    accuracy_score(y_test, log_predictions),
    accuracy_score(y_test, knn_predictions),
    accuracy_score(y_test, tree_predictions),
    accuracy_score(y_test, rf_predictions),
    accuracy_score(y_test, svm_predictions),
    accuracy_score(y_test, gb_predictions)
]

plt.figure()
plt.bar(models, scores)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()
# 📊 Feature Importance (Random Forest)
importance = rf.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importance)
plt.title("Feature Importance")
plt.show()
# --------------------------------
# STEP 7: Deployment Simulation
# --------------------------------
joblib.dump(log_model_tuned, "customer_loyalty_model.pkl")
print("\n7) Model saved as customer_loyalty_model.pkl")

loaded_model = joblib.load("customer_loyalty_model.pkl")


# --------------------------------
# STEP 8: Real-Time Prediction
# --------------------------------
new_customer = pd.DataFrame([[1000, 5]], columns=["MonthlySpend", "PurchaseFrequency"])

deployment_prediction = loaded_model.predict(new_customer)

print("\n8) Deployment Prediction:")
if deployment_prediction[0] == 1:
    print("Customer is likely to be a LOYAL customer")
else:
    print("Customer is NOT likely to be a loyal customer")

print("\n--- End of Industry ML Capstone Project ---\n")