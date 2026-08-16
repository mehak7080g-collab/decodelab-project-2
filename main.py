from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and understand the dataset
iris = load_iris()

X = iris.data
y = iris.target

print("=" * 60)
print("        PROJECT 2: DATA CLASSIFICATION USING AI")
print("=" * 60)
print(f"Dataset: Iris")
print(f"Number of samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Classes: {', '.join(iris.target_names)}")

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Apply a simple classification algorithm
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 5. Test the trained model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel: Logistic Regression")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# 6. Classify a new sample
new_sample = [[5.1, 3.5, 1.4, 0.2]]
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)[0]

print("New Sample Prediction:")
print(f"Input: {new_sample[0]}")
print(f"Predicted class: {iris.target_names[prediction]}")

print("\nProject completed successfully.")
