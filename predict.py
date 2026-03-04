import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# ---------- Streamlit basic settings ----------
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.write("")  # small spacing

# ---------- File upload ----------
st.sidebar.header("1️⃣ Upload Stock CSV File")
uploaded_file = st.sidebar.file_uploader("Upload .csv", type=["csv"])

if uploaded_file is None:
    st.title("📈 Stock Price Prediction")
    st.info("Please upload your **Stock CSV file** to continue.")
    st.stop()

# ---------- Get filename to use as heading ----------
file_name = uploaded_file.name.rsplit(".", 1)[0]  # remove .csv
clean_title = file_name.replace("_", " ")         # nicer spacing

st.title(f"📈 {clean_title} Stock Prediction")

# ---------- Read CSV ----------
df = pd.read_csv(uploaded_file)

# ---------- Clean & Rename Columns ----------
df.columns = df.columns.str.strip()

df.rename(
    columns={"OPEN": "Open","HIGH": "High","LOW": "Low","close": "Close","CLOSE": "Close","VOLUME": "Volume","Date": "Date"},inplace=True)

# ---------- Validate Required Columns ----------
required = ["Date", "Open", "High", "Low", "Close", "Volume"]
missing = [col for col in required if col not in df.columns]

if missing:
    st.error(f"❌ Missing required columns: {missing}")
    st.stop()

# ---------- Convert Date ----------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

# ---------- Auto Date Range Text (Jan 2024 – Dec 2024 style) ----------
start_date_csv = df["Date"].min()
end_date_csv = df["Date"].max()

start_str = start_date_csv.strftime("%b %Y")   # Example → Jan 2024
end_str = end_date_csv.strftime("%b %Y")       # Example → Dec 2024

st.write(f"{start_str} – {end_str} Data | Model: Trained ML Regressor")

# ---------- Convert Numeric Columns ----------
numeric_cols = ["Open", "High", "Low", "Close", "Volume"]

for col in numeric_cols:
    df[col] = df[col].astype(str).str.replace(",", "", regex=False)

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=numeric_cols)

# ---------- Feature Engineering ----------
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

df["HL_Perc"] = (df["High"] - df["Low"]) / df["Low"] * 100
# Match notebook: (Close - Open) / Open * 100
df["OC_Perc"] = (df["Close"] - df["Open"]) / df["Open"] * 100

# ---------- Load Model, Scaler & Feature Columns ----------
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
except Exception as e:
    st.error(
        "❌ Missing or invalid model files.\n\n"
        "Make sure **model.pkl**, **scaler.pkl**, and **feature_cols.pkl** "
        "are in the same folder as this Streamlit app."
    )
    st.stop()
# ---------- Show Which Model Is Used ----------
model_name = type(model).__name__  # Example: RandomForestRegressor
st.subheader("🤖 Model Used for Prediction")
st.success(f"Current trained model: **{model_name}**")

# ---------- Build X, y using the same feature_cols as training ----------
for col in feature_cols:
    if col not in df.columns:
        st.error(f"❌ Required feature column from training not found in CSV: `{col}`")
        st.stop()

X = df[feature_cols]
y = df["Close"]

# ---------- Predict on all historical data ----------
X_scaled = scaler.transform(X)
y_pred_all = model.predict(X_scaled)

mse = mean_squared_error(y, y_pred_all)
r2 = r2_score(y, y_pred_all)

# ---------- Show Metrics ----------
st.subheader("📊 Model Performance (on uploaded data)")
col1, col2 = st.columns(2)
col1.metric("R² Score", f"{r2:.4f}")
col2.metric("MSE", f"{mse:.2f}")

# ---------- Actual vs Predicted ----------
st.subheader(f"🟦 Actual vs 🟧 Predicted Close Price – {clean_title}")
results_df = pd.DataFrame({"Date": df["Date"],"Actual_Close": y.values,"Predicted_Close": y_pred_all,}).sort_values("Date").set_index("Date")

st.line_chart(results_df[["Actual_Close", "Predicted_Close"]])

# ---------- Close Price Trend ----------
with st.expander(f"📉 Close Price Trend – {clean_title}"):
    trend_df = df[["Date", "Close"]].set_index("Date")
    st.line_chart(trend_df["Close"])

# ---------- Future Prediction ----------
st.subheader(f"🔮 Future Close Price Prediction – {clean_title}")

col_start, col_end = st.columns(2)
start_date = col_start.date_input("Start Date", date(2025, 1, 1))
end_date = col_end.date_input("End Date", date(2025, 12, 30))

if start_date > end_date:
    st.error("⚠️ End date must be AFTER start date.")
else:
    future_dates = pd.date_range(start=start_date, end=end_date)
    future_df = pd.DataFrame({"Date": future_dates})

    # Date-based features
    future_df["Day"] = future_df["Date"].dt.day
    future_df["Month"] = future_df["Date"].dt.month
    future_df["Year"] = future_df["Date"].dt.year

    # Use last known data for other features (same assumption as notebook)
    last = df.iloc[-1]

    future_df["Open"] = last["Open"]
    future_df["High"] = last["High"]
    future_df["Low"] = last["Low"]
    future_df["Volume"] = last["Volume"]
    future_df["HL_Perc"] = last["HL_Perc"]
    future_df["OC_Perc"] = last["OC_Perc"]

    # Predict
    future_scaled = scaler.transform(future_df[feature_cols])
    future_df["Predicted_Close"] = model.predict(future_scaled)

    st.write(f"### 📅 Future Predicted Close Prices – {clean_title}")
    st.dataframe(future_df[["Date", "Predicted_Close"]].set_index("Date"))

    st.write(f"### 📈 Future Price Trend – {clean_title}")
    st.line_chart(future_df.set_index("Date")["Predicted_Close"])

    # ---------- Download CSV ----------
    st.download_button(
        "Download Future Predictions CSV",
        future_df[["Date", "Predicted_Close"]].to_csv(index=False),
        file_name=f"{clean_title}_future_predictions.csv",
        mime="text/csv",
    )
