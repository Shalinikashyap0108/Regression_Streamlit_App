# app.py

import streamlit as st
import pandas as pd
from models import train_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm

st.set_page_config(page_title="Regression Model App", layout="wide")

st.title("🏡 Housing Price Regression App")

# --- Upload dataset ---
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Ames_Housing_Subset.csv")

st.write("### Dataset Preview", df.head())

# --- Feature Selection ---
target = st.sidebar.selectbox("Select Target Variable", df.columns)
features = st.sidebar.multiselect("Select Feature Columns", [col for col in df.columns if col != target])

if not features:
    st.warning("Please select at least one feature.")
    st.stop()

X = df[features]
y = df[target]

# --- Train/Test Split ---
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

# --- Model Selection ---
model_name = st.sidebar.selectbox("Choose Model", ["Linear Regression", "Decision Tree", "Random Forest"])

# --- Hyperparameters ---
params = {}
if model_name == "Decision Tree":
    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 5)
elif model_name == "Random Forest":
    params["n_estimators"] = st.sidebar.slider("Number of Trees", 10, 200, 100)
    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 5)

# --- Train Model ---
if st.sidebar.button("Train Model"):
    model = train_model(model_name, X_train, y_train, params)
    y_pred = model.predict(X_test)

    st.subheader("📊 Model Evaluation")
    st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

    if hasattr(model, 'feature_importances_'):
        st.subheader("🔍 Feature Importances")
        importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        st.bar_chart(importance)

    if model_name == "Linear Regression":
        st.subheader("📈 Coefficients")
        coef = pd.Series(model.coef_, index=features)
        st.write(coef)

        st.subheader("📘 Statistical Summary")
        X_train_const = sm.add_constant(X_train)
        ols = sm.OLS(y_train, X_train_const).fit()
        st.text(ols.summary())

    st.subheader("📉 Residual Plot")
    residuals = y_test - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    st.pyplot(fig)
