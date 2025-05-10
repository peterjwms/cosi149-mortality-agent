from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
from xgboost import XGBClassifier



def predict_mortality(input_csv):
	# Load the trained model
	model = joblib.load('mortality_prediction_model.pkl')
	print("Model loaded successfully.")

	# Load the input data
	input_data = pd.read_csv(input_csv)

	# Make predictions
	predictions = model.predict(input_data)
	return predictions

# print(predict_mortality('AI_agent_test_sepsis_features.csv'))


def evaluate_test_set(X_test, y_test, model):
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print(f"Test Set Accuracy: {accuracy:.4f}")
	precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
	print(f"Test Set Precision: {precision:.4f}")
	print(f"Test Set Recall: {recall:.4f}")
	print(f"Test Set F1 Score: {f1:.4f}")


def train_model(model, X_train, y_train):
	# Train the model
	model.fit(X_train, y_train)
	print("Model training completed.")
	return model


def save_model(model, filename):
	# Save the model
	joblib.dump(model, filename)
	print(f"Model saved as '{filename}'")


def load_model(filename):
	# Load the model
	model = joblib.load(filename)
	print("Model loaded successfully.")
	return model


def feature_importance(model, X_train):
	# Get feature importance
	if hasattr(model, 'feature_importances_'):
		importances = model.feature_importances_
	else:
		importances = model.get_booster().get_score(importance_type='weight')
	
	# Create a DataFrame for feature importance
	feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
	feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
	print(feature_importance_df)
	feature_importance_df.to_csv(f'{model.__class__.__name__}_feature_importance2.csv', index=False)


if __name__ == "__main__":
	do_training = False
	# Load the dataset
	data = pd.read_csv('AI_agent_train_sepsis.csv')

	# Prepare the data
	X = data.drop(columns=['mortality_90d', 'icustayid', 'charttime'])  # Features
	y = data['mortality_90d']  # Target variable

	# Split the data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	models = [
		DecisionTreeClassifier(random_state=42),
		XGBClassifier(random_state=42),
		RandomForestClassifier(random_state=42)
	]
	if do_training:
		for model in models:
			print(f"Training model: {model.__class__.__name__}")
			model = train_model(model, X_train, y_train)
			save_model(model, f'mortality_models/{model.__class__.__name__}_mortality_prediction_model2.pkl')

			# Evaluate the model
			evaluate_test_set(X_test, y_test, model)
			feature_importance(model, X_train)
	else:
		# model = load_model('mortality_models/RandomForestClassifier_mortality_prediction_model.pkl')
		for model in glob('mortality_models/*2.pkl'):
			model = load_model(model)
			print(f"Evaluating model: {model.__class__.__name__}")
			
			evaluate_test_set(X_test, y_test, model)
		# feature_importance(model, X_train)