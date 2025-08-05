import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Breast Cancer Detection", layout="centered")

# Load dataset
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.feature_names, data.target_names

df, feature_names, target_names = load_data()

st.title("🧠 Breast Cancer Detection using ML")
st.markdown("Predict whether a tumor is **Benign** or **Malignant** based on cell features.")

# Sidebar input for prediction
st.sidebar.header("🔍 Input Features for Prediction")
def user_input_features():
    inputs = {}
    for feature in feature_names:
        inputs[feature] = st.sidebar.slider(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    return pd.DataFrame([inputs])

input_df = user_input_features()

# Data preparation
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
input_scaled = scaler.transform(input_df)

# Model selection
model_option = st.selectbox("Select Classifier", ("Logistic Regression", "SVM", "Random Forest"))

if model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_option == "SVM":
    model = SVC(probability=True)
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Show results
st.subheader("📊 Model Performance")
st.write(f"**Accuracy:** {accuracy*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Classification report
st.text("Classification Report:")
st.code(classification_report(y_test, y_pred, target_names=target_names))

# Prediction on sidebar input
prediction = model.predict(input_scaled)[0]
proba = model.predict_proba(input_scaled)[0]

st.subheader("🔮 Prediction on Your Input")
st.write(f"**Prediction:** {'Benign' if prediction == 1 else 'Malignant'}")
st.write(f"**Probability:** Benign: {proba[1]*100:.2f}%, Malignant: {proba[0]*100:.2f}%")

st.markdown("---")
st.caption("Built with Streamlit · Dataset: Breast Cancer Wisconsin (Diagnostic)")
