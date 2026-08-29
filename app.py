import streamlit as st
import pandas as pd
import joblib
from sklearn.cluster import KMeans

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Intelligence System",
    page_icon="🛒",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🛒 Customer Intelligence System")
st.subheader("Machine Learning Dashboard for Customer Segmentation & Loyalty Prediction")

st.write(
    "Analyze customer behavior using K-Means clustering and "
    "Machine Learning-based loyalty prediction."
)

# -----------------------------
# Create Customer Dataset
# -----------------------------
data = {
    "CustomerID": range(1, 21),
    "MonthlySpend": [
        500, 800, 200, 1200, 1500, 300, 700, 1600, 400, 1000,
        1100, 600, 900, 1300, 1400, 350, 750, 1550, 450, 1050
    ],
    "PurchaseFrequency": [
        2, 5, 1, 6, 7, 1, 4, 8, 2, 5,
        6, 3, 5, 7, 6, 2, 4, 8, 2, 5
    ],
    "AvgOrderValue": [
        250, 300, 200, 400, 450, 150, 280, 500, 220, 350,
        370, 260, 310, 420, 430, 180, 290, 480, 210, 360
    ],
    "TenureMonths": [
        12, 24, 6, 36, 40, 8, 20, 45, 10, 30,
        32, 18, 25, 38, 42, 9, 22, 44, 11, 28
    ],
    "DiscountUsage": [
        0.2, 0.5, 0.1, 0.6, 0.7, 0.2, 0.4, 0.8, 0.3, 0.5,
        0.6, 0.3, 0.5, 0.7, 0.6, 0.2, 0.4, 0.8, 0.3, 0.5
    ],
    "Returns": [
        1, 2, 0, 3, 2, 1, 1, 4, 0, 2,
        2, 1, 2, 3, 2, 1, 1, 4, 0, 2
    ],
    "LoyalCustomer": [
        0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
        1, 0, 1, 1, 1, 0, 0, 1, 0, 1
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# K-Means Customer Segmentation
# -----------------------------
X_cluster = df[["MonthlySpend", "PurchaseFrequency"]]

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["CustomerSegment"] = kmeans.fit_predict(X_cluster)

# -----------------------------
# Load Trained Model
# -----------------------------
try:
    model = joblib.load("customer_loyalty_model.pkl")
except Exception as e:
    st.error("Unable to load customer_loyalty_model.pkl")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("👤 Customer Information")

monthly_spend = st.sidebar.number_input(
    "Monthly Spend (₹)",
    min_value=0,
    max_value=100000,
    value=1000,
    step=100
)

purchase_frequency = st.sidebar.number_input(
    "Purchase Frequency",
    min_value=0,
    max_value=100,
    value=5,
    step=1
)

predict_button = st.sidebar.button(
    "🔮 Predict Customer"
)

# -----------------------------
# Dashboard Metrics
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Customers",
        len(df)
    )

with col2:
    st.metric(
        "Average Monthly Spend",
        f"₹{df['MonthlySpend'].mean():.0f}"
    )

with col3:
    st.metric(
        "Loyal Customers",
        int(df["LoyalCustomer"].sum())
    )

st.divider()

# -----------------------------
# Customer Segmentation
# -----------------------------
st.header("👥 Customer Segmentation")

st.write(
    "Customers are grouped into three segments using K-Means "
    "based on monthly spending and purchase frequency."
)

st.scatter_chart(
    df,
    x="MonthlySpend",
    y="PurchaseFrequency",
    color="CustomerSegment"
)

st.dataframe(
    df[
        [
            "CustomerID",
            "MonthlySpend",
            "PurchaseFrequency",
            "CustomerSegment"
        ]
    ],
    use_container_width=True
)

st.divider()

# -----------------------------
# Customer Loyalty Prediction
# -----------------------------
st.header("❤️ Customer Loyalty Prediction")

st.write(
    "Enter customer behavior in the sidebar and click "
    "**Predict Customer**."
)

if predict_button:

    new_customer = pd.DataFrame(
        [[monthly_spend, purchase_frequency]],
        columns=[
            "MonthlySpend",
            "PurchaseFrequency"
        ]
    )

    prediction = model.predict(new_customer)

    if prediction[0] == 1:
        st.success(
            "🟢 This customer is likely to be a LOYAL customer."
        )
    else:
        st.warning(
            "🟠 This customer is NOT likely to be a loyal customer."
        )

    # Determine nearest customer segment
    segment = kmeans.predict(
        new_customer[
            ["MonthlySpend", "PurchaseFrequency"]
        ]
    )[0]

    st.info(
        f"📊 Predicted Customer Segment: **Segment {segment}**"
    )

# -----------------------------
# Project Information
# -----------------------------
st.divider()

st.header("📌 Project Overview")

st.write(
    """
    **Customer Intelligence System** uses Machine Learning to analyze
    customer behavior and predict customer loyalty.

    **Techniques Used:**
    - K-Means Clustering
    - Logistic Regression
    - Customer Segmentation
    - Loyalty Prediction
    - Real-Time Prediction

    **Technologies:**
    Python, Pandas, NumPy, Scikit-learn, Joblib, Streamlit
    """
)

st.caption(
    "Customer Intelligence System | Machine Learning Project"
)
