import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv(r"C:\Users\Admin\Desktop\MLPRACTICE\DATASET\heart.csv")

# Feature and target separation
x = df[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"]]  # Features
y = df["target"]  # Target variable

# Split the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=42)

# Initialize and train the Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(xtrain, ytrain)

# Predict on the test set
ypred = clf.predict(xtest)

# Calculate accuracy
accuracy = accuracy_score(ytest, ypred)
print("Accuracy:", accuracy)

# Save the trained model to a file using pickle
with open(r'C:\Users\Admin\Desktop\MLPRACTICE\Heart_disease_Predict_Project\heartmodel.pkl', 'wb') as file:
    pickle.dump(clf, file)

print("Model saved successfully.")
