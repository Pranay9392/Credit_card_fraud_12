import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load dataset for training the model
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
balanced_data = pd.concat([legit_sample, fraud], axis=0)

# Split data into features and target
X = balanced_data.drop(columns="Class", axis=1)
y = balanced_data["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

# Streamlit App
st.title("📊 Student Credit Card Fraud Detector 😎")

st.write("Hey there! Let's detect fraud using our very own machine learning model! 🔍")

st.write("### Model Performance Stats 🧠")
st.write(f"Training Accuracy: **{train_acc * 100:.2f}%**")
st.write(f"Test Accuracy: **{test_acc * 100:.2f}%**")

# Section to upload CSV
st.write("### Upload a CSV File Containing Transaction Data 📂")
uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

# Check button to display the data and proceed with prediction
if uploaded_file is not None:
    # Read the uploaded CSV file
    input_data = pd.read_csv(uploaded_file)

    # Show a "Check" button to review the data before making predictions
    if st.button("Check Data 📋"):
        st.write("### Here's the data you uploaded:")
        st.write(input_data)

    # Ensure that the uploaded file has the correct number of features
    if set(X.columns) == set(input_data.columns):
        # Add a second button for prediction after displaying data
        if st.button("Get Predictions ✅"):
            # Make predictions for all the rows in the uploaded file
            predictions = model.predict(input_data)

            # Add predictions to the uploaded file
            input_data['Prediction'] = predictions

            # Change the wording to more friendly terms
            input_data['Prediction Result'] = input_data['Prediction'].apply(lambda x: "Legitimate Transaction 🟢" if x == 0 else "Fraudulent Transaction 🔴")

            # Display the results
            st.write("### Fraud Detection Results:")
            st.write(input_data[['Prediction Result']])

            # Download option for the output file with predictions
            csv = input_data.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV 📥",
                data=csv,
                file_name='fraud_detection_predictions.csv',
                mime='text/csv',
            )
    else:
        st.error("The uploaded file does not match the required features. Please double-check your file format.")
else:
    st.write("🛠️ Please upload a CSV file to start the process.")
