import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the non-anomaly data
non_anomaly_csv_filename = 'non_anomaly_data.csv'
non_anomaly_df = pd.read_csv(non_anomaly_csv_filename)

# Open the Mitos Spreadsheet file
#st.write("Opening Mitos Spreadsheet file...")
#st.csv_open("non_anomaly_data.csv")

# Display the first sheet
#st.write(st.get_active_sheet().name)

# Display the first row of the first sheet
#st.write(st.get_active_sheet().rows[0])

# Load the Isolation Forest model
model_filename = "IsolationForest.joblib"
isolation_forest = joblib.load(model_filename)

# Load the StandardScaler
scaler_filename = "StandardScaler.joblib"
scaler = joblib.load(scaler_filename)

st.title("Anomaly Detection App with Isolation Forest")

st.sidebar.title("Input Feature Values")
transaction_dollar_amount = st.sidebar.slider("Transaction Dollar Amount", min_value=0.0, max_value=10000.0)
longitude = st.sidebar.slider("Longitude (Long)", min_value=-180.0, max_value=180.0)
latitude = st.sidebar.slider("Latitude (Lat)", min_value=-90.0, max_value=90.0)
credit_card_limit = st.sidebar.slider("Credit Card Limit", min_value=0, max_value=50000)
year = st.sidebar.slider("Year", min_value=2000, max_value=2030)
month = st.sidebar.slider("Month", min_value=1, max_value=12)
day = st.sidebar.slider("Day", min_value=1, max_value=31)

submitted = st.sidebar.button("Submit")

if submitted:
    input_data = {
        'transaction_dollar_amount': transaction_dollar_amount,
        'Long': longitude,
        'Lat': latitude,
        'credit_card_limit': credit_card_limit,
        'year': year,
        'month': month,
        'day': day
    }

    selected_columns = pd.DataFrame([input_data])

    # Standardize the input data using the loaded StandardScaler
    selected_columns_scaled = scaler.transform(selected_columns)

    # Apply Isolation Forest for anomaly detection on the non-anomaly dataset
    non_anomaly_scores = isolation_forest.decision_function(scaler.transform(non_anomaly_df))

# Apply Isolation Forest for anomaly detection on your single input data
    your_anomaly_score = isolation_forest.decision_function(selected_columns_scaled)[0]



    # Calculate the minimum and maximum anomaly scores from non-anomaly data
    min_non_anomaly_score = np.min(non_anomaly_scores)
    max_non_anomaly_score = np.max(non_anomaly_scores)

# Add a margin of error for the range
    margin = 0.5
    min_threshold = min_non_anomaly_score - margin
    max_threshold = max_non_anomaly_score + margin

    # Determine if the input data point is an anomaly based on the score
    #is_anomaly = your_anomaly_score >= np.percentile(non_anomaly_scores, 95)

    # Determine if the input data point is an anomaly based on the score
    is_anomaly = your_anomaly_score < min_threshold or your_anomaly_score > max_threshold


# Print the anomaly status
    st.subheader("Anomaly Classification")
    if is_anomaly:
        st.write("Prediction Result: 🚨 Anomaly Detected!")
    else:
        st.write("Prediction Result: ✅ Not Anomaly")

# Create a bar plot to visualize the anomaly score distribution and your data point's score
    plt.figure(figsize=(8, 5))

# Plot the distribution of anomaly scores from the non-anomaly dataset
    sns.histplot(non_anomaly_scores, kde=True, color='gray', label='Non-Anomaly Score Distribution')

# Plot your data point's anomaly score
    plt.axvline(x=your_anomaly_score, color='blue', linestyle='dashed', label='Your Data Point')

# Set labels and title
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Anomaly Score Distribution and Your Data Point')
    plt.legend()
#plt.grid(True)

# Display the histogram plot
    st.pyplot(plt)


# Explain the results
    st.write("The input data point has been classified as an anomaly." if is_anomaly
            else "The input data point is not classified as an anomaly.")
    st.write("The anomaly score is:", your_anomaly_score)
    st.write("The threshold for anomaly detection is:", min_threshold, "to", max_threshold)

    # Create a scatter plot for longitude and latitude
    fig, ax = plt.subplots(figsize=(10, 8))

# Plot non-anomaly data
    sns.scatterplot(data=non_anomaly_df, x='Long', y='Lat', color='lightgrey', label='Normal 🏙️', ax=ax)

# Plot input data
    if is_anomaly:
        ax.scatter(selected_columns['Long'], selected_columns['Lat'], color='red', label='Suspicious 🚩', s=100, marker='x')
        anomaly_marker = 'Suspicious 🚩'
    else:
        ax.scatter(selected_columns['Long'], selected_columns['Lat'], color='green', label='Valid ✅', s=100, marker='o')
        anomaly_marker = 'Valid ✅'

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Location Plot: Anomaly Detection 🗺️")
    ax.legend()
    ax.grid(True)

# Show the scatter plot in Streamlit
    st.subheader("Location Plot: Anomaly Detection 🗺️")
    st.pyplot(fig)

# Explanation based on the anomaly classification
    st.subheader("Anomaly Classification")
    if your_anomaly_score < min_threshold or your_anomaly_score > max_threshold:
        st.write("Prediction Result: 🚨 Anomaly Detected!")
    else:
        st.write("Prediction Result: ✅ Not Anomaly")

# Explain the results
    # Explain the results
    st.write("The location plot visualizes the anomaly detection result based on longitude and latitude.")
    if your_anomaly_score < min_threshold or your_anomaly_score > max_threshold:
        st.write("The input data point is marked as Suspicious 🚩 due to its anomaly score.")
        st.write("The red 'x' marker indicates a suspicious location.")
    else:
        st.write("The input data point is marked as Valid ✅ due to its anomaly score.")
        st.write("The green 'o' marker indicates a valid location.")
