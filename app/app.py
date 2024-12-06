import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
data = pd.read_csv('data.csv')

# App Title and Sidebar
st.title("Diabetes Prediction and Analysis")
st.sidebar.title("Navigation")

# Navigation Options
menu = st.sidebar.radio("Choose an option:", ["Show Dataset", "Visualizations", "Predict Diabetes", "View Predictions"])

# Initialize a DataFrame to store predictions (in-memory)
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = pd.DataFrame(columns=[
        'Age', 'Hypertension', 'Heart Disease', 'BMI', 'HbA1c Level', 'Blood Glucose Level', 'Prediction'
    ])

# Display Dataset
if menu == "Show Dataset":
    st.subheader("Dataset")
    st.write(data)
    st.write("Shape of the dataset:", data.shape)

# Visualizations
if menu == "Visualizations":
    st.subheader("Visualizations")

    # Age Distribution
    st.write("Age Distribution")
    fig, ax = plt.subplots()
    data['age'].hist(bins=20, ax=ax, color='skyblue')
    st.pyplot(fig)

    # BMI vs Diabetes
    st.write("BMI vs Diabetes")
    fig, ax = plt.subplots()
    data[data['diabetes'] == 1]['bmi'].hist(bins=20, ax=ax, alpha=0.7, label='Diabetic', color='red')
    data[data['diabetes'] == 0]['bmi'].hist(bins=20, ax=ax, alpha=0.7, label='Non-Diabetic', color='green')
    ax.legend()
    st.pyplot(fig)

# Predict Diabetes
if menu == "Predict Diabetes":
    st.subheader("Predict Diabetes")

    # Input Fields
    age = st.number_input("Age", 1, 100, step=1)
    hypertension = st.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
    heart_disease = st.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1])
    bmi = st.number_input("BMI", 0.0, 50.0, step=0.1)
    hba1c = st.number_input("HbA1c Level", 0.0, 15.0, step=0.1)
    glucose = st.number_input("Blood Glucose Level", 0, 500, step=1)

    # Train Model
    X = data[['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
    y = data['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make Prediction
    if st.button("Predict"):
        input_features = pd.DataFrame([{
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'bmi': bmi,
            'HbA1c_level': hba1c,
            'blood_glucose_level': glucose
        }])
        prediction = model.predict(input_features)[0]
        st.write("Prediction (1=Diabetic, 0=Non-Diabetic):", prediction)

        # Save Prediction
        new_record = {
            'Age': age,
            'Hypertension': hypertension,
            'Heart Disease': heart_disease,
            'BMI': bmi,
            'HbA1c Level': hba1c,
            'Blood Glucose Level': glucose,
            'Prediction': prediction
        }
        st.session_state['predictions'] = pd.concat(
            [st.session_state['predictions'], pd.DataFrame([new_record])],
            ignore_index=True
        )
        st.success("Prediction recorded successfully!")

# View Predictions
if menu == "View Predictions":
    st.subheader("Patient Predictions")
    if not st.session_state['predictions'].empty:
        st.dataframe(st.session_state['predictions'])

        # Option to Save to CSV
        if st.button("Download Predictions as CSV"):
            st.session_state['predictions'].to_csv('predictions.csv', index=False)
            st.success("Predictions saved to predictions.csv!")
    else:
        st.write("No predictions recorded yet.")