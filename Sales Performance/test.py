import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load('linear_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature names and mappings
feature_names = ['Age', 'Gender', 'EducationLevel', 'SalesTraining', 'PreviousSalesPerformance']
gender_mapping = {'Male': 0, 'Female': 1}
education_level_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
sales_training_mapping = {'No': 0, 'Yes': 1}

def predict_sales_performance(model, scaler, feature_names):
    # Prompt user for input
    print("Please enter the following details:")

    # Collect inputs from user
    user_inputs = []
    for feature in feature_names:
        while True:
            try:
                if feature == 'Gender':
                    value = input(f"{feature} (Male/Female): ").strip()
                    if value not in gender_mapping:
                        raise ValueError
                    user_inputs.append(gender_mapping[value])
                elif feature == 'SalesTraining':
                    value = input(f"{feature} (Yes/No): ").strip()
                    if value not in sales_training_mapping:
                        raise ValueError
                    user_inputs.append(sales_training_mapping[value])
                elif feature == 'EducationLevel':
                    value = input(f"{feature} (High School/Bachelor/Master/PhD): ").strip()
                    if value not in education_level_mapping:
                        raise ValueError
                    user_inputs.append(education_level_mapping[value])
                else:
                    value = float(input(f"{feature}: ").strip())
                    if feature == 'Age' and (value < 18 or value > 100):  # Example validation for age
                        raise ValueError
                    if feature == 'PreviousSalesPerformance' and value < 0:  # Example validation
                        raise ValueError
                    user_inputs.append(value)
                break
            except ValueError:
                print(f"Invalid input. Please enter a valid value for {feature}.")

    # Print collected user inputs for debugging
    print(f"User Inputs: {user_inputs}")
    
    # Check the number of inputs
    if len(user_inputs) != len(feature_names):
        raise ValueError("Mismatch between number of features and inputs provided.")
    
    # Convert user inputs to DataFrame with correct column names
    user_input_df = pd.DataFrame([user_inputs], columns=feature_names)

    # Preprocess input features
    user_input_scaled = scaler.transform(user_input_df)

    # Make prediction
    prediction = model.predict(user_input_scaled)
    
    # Extract the predicted value from the array
    predicted_value = prediction[0]  # Assuming the model returns a single value

    # Show the results
    print("\nPrediction Results:")
    print(f"Estimated Sales Performance: {predicted_value:.2f}")

# Example usage
predict_sales_performance(model, scaler, feature_names)
