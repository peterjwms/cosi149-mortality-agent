import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib


# Load the dataset
data = pd.read_csv('AI_agent_train_sepsis.csv')

# Prepare the data
X = data.drop(columns=['mortality_90d'])  # Features
y = data['mortality_90d']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, 'mortality_prediction_model.pkl')
print("Model saved as 'mortality_prediction_model.pkl'")

def predict_mortality(input_csv):
	# Load the trained model
	model = joblib.load('mortality_prediction_model.pkl')
	print("Model loaded successfully.")

	# Load the input data
	input_data = pd.read_csv(input_csv)

	# Make predictions
	predictions = model.predict(input_data)
	return predictions

print(predict_mortality('AI_agent_test_sepsis_features.csv'))