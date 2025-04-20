import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(page_title="Production Efficiency Simulator", layout="wide")
st.title("🏭 Production Efficiency Simulator Dashboard")

# Load dataset
df = pd.read_csv("production_efficiency_simulator_dataset.csv")
df["Production_Date"] = pd.to_datetime(df["Production_Date"])

# Sidebar filters
st.sidebar.header("🔍 Filter Data")
date_range = st.sidebar.date_input("Select Date Range", [df["Production_Date"].min(), df["Production_Date"].max()])
selected_shifts = st.sidebar.multiselect("Select Shifts", df["Shift"].unique(), df["Shift"].unique())

filtered_df = df[(df["Production_Date"] >= pd.to_datetime(date_range[0])) &
                 (df["Production_Date"] <= pd.to_datetime(date_range[1])) &
                 (df["Shift"].isin(selected_shifts))]

# 🎯 Scenario Planner
df_section = st.sidebar.header("🧠 Scenario Planner — What If Analysis")
sim_defect = st.sidebar.slider("Set Simulated Defect Rate (%)", 0.0, 20.0, 5.0)
sim_downtime = st.sidebar.slider("Set Simulated Downtime (minutes)", 0, 180, 30)
sim_material_shortage = st.sidebar.slider("Set Simulated Material Shortage (%)", 0.0, 15.0, 3.0)

# Adjust dataset based on user inputs (Defect, Downtime, Material Shortage)
simulated_df = filtered_df.copy()

simulated_df["Actual_Units"] = [
    max(
        0,
        planned 
        - int((sim_downtime / 60) * planned * 0.2) 
        - int(planned * sim_defect / 100) 
        - int(planned * sim_material_shortage / 100)  # Material Shortage Impact
    )
    for planned in simulated_df["Planned_Units"]
]

simulated_df["Efficiency_%"] = np.round(
    (simulated_df["Actual_Units"] / simulated_df["Planned_Units"]) * 100, 2
)

simulated_df["Efficiency_%"] = np.round((simulated_df["Actual_Units"] / simulated_df["Planned_Units"]) * 100, 2)

# Calculate the average defect rate for dynamic background
average_defect_rate = simulated_df["Defect_Rate_%"].mean()

# Set background color based on defect level
if average_defect_rate < 3:
    background_color = "#e8f5e9"  # Soft Green for Stable Production
elif average_defect_rate < 6:
    background_color = "#fffde7"  # Light Yellow for Attention Needed
else:
    background_color = "#ffebee"  # Light Red for High Defects - Critical

# Inject the background color into the entire app
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {background_color};
        transition: background-color 1s ease;
    }}
    h1, h2 {{
        color: #003366;
    }}
    </style>
    """,
    unsafe_allow_html=True
)



# Calculate average defect rate for background color logic
average_defect_rate = simulated_df["Defect_Rate_%"].mean()

# Choose background color based on defect severity
if average_defect_rate < 3:
    bg_color = "#e8f5e9"  # light green
elif average_defect_rate < 6:
    bg_color = "#fff8e1"  # light yellow
else:
    bg_color = "#ffebee"  # light red

# Inject dynamic background style into Streamlit app
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg_color};
    }}
    h1, h2 {{
        color: #003366;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# KPI overview
st.subheader("📊 Scenario-Based KPI Results")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Planned Units", f"{simulated_df['Planned_Units'].sum()}")
col2.metric("Actual Units", f"{simulated_df['Actual_Units'].sum()}")
col3.metric("Avg Efficiency", f"{simulated_df['Efficiency_%'].mean():.2f}%")
col4.metric("Simulated Downtime", f"{sim_downtime * len(simulated_df)} mins")
col5.metric("Simulated Defect Rate", f"{sim_defect:.2f}%")

st.markdown("---")

# 📊 Shift Performance Comparison
st.subheader("📊 Shift-wise Average Efficiency Comparison")
shift_performance = simulated_df.groupby("Shift")["Efficiency_%"].mean().reset_index()
fig_shift = px.bar(
    shift_performance, x="Shift", y="Efficiency_%",
    color="Efficiency_%", color_continuous_scale="Viridis",
    title="Average Efficiency by Shift"
)
st.plotly_chart(fig_shift, use_container_width=True)

st.markdown("---")

# 📈 Efficiency Trend Over Time
st.subheader("📈 Efficiency Trend Over Time (Simulated Data)")
efficiency_trend = simulated_df.groupby("Production_Date")["Efficiency_%"].mean().reset_index()
fig1 = px.line(efficiency_trend, x="Production_Date", y="Efficiency_%", title="Average Efficiency Over Time")
st.plotly_chart(fig1, use_container_width=True)


# 🧯 Downtime Impact on Production
st.subheader("🧯 Downtime Impact on Production (Simulated Data)")
fig2 = px.scatter(simulated_df, x="Downtime_Minutes", y="Actual_Units", trendline="ols",
                  title="Downtime vs. Actual Production")
st.plotly_chart(fig2, use_container_width=True)

# 📊 Defect Rate Distribution - Smoothed KDE Plot
st.subheader("📊 Defect Rate Distribution (Simulated Data)")
fig, ax = plt.subplots(figsize=(8, 4))
sns.kdeplot(simulated_df["Defect_Rate_%"], fill=True, color="skyblue", linewidth=2, ax=ax)
ax.set_title("Defect Rate Distribution Curve")
ax.set_xlabel("Defect Rate (%)")
ax.set_ylabel("Density")
st.pyplot(fig)

st.markdown("---")


# 🚨 Anomaly Detection
st.subheader("🚨 Anomaly Detection: Low Efficiency Alerts")
thresh = 60
anomalies = simulated_df[simulated_df["Efficiency_%"] < thresh]
if not anomalies.empty:
    st.warning(f"{len(anomalies)} days found where efficiency fell below {thresh}%. Review suggested.")
    st.dataframe(anomalies[["Production_Date", "Shift", "Efficiency_%"]].sort_values(by="Efficiency_%"))
else:
    st.success("All production days are operating above efficiency threshold!")

st.markdown("---")

# 🔮 Predictive Output Planner
st.subheader("🔮 Predictive Output Planner: Next 7 Days")
historical = simulated_df.groupby("Production_Date")["Actual_Units"].sum().reset_index()
historical["Days"] = (historical["Production_Date"] - historical["Production_Date"].min()).dt.days
if len(historical) > 7:
    X = historical[["Days"]]
    y = historical["Actual_Units"]
    model = LinearRegression().fit(X, y)
    future_days = pd.DataFrame({"Days": [X["Days"].max() + i for i in range(1, 8)]})
    future_predictions = model.predict(future_days)
    future_dates = pd.date_range(historical["Production_Date"].max() + pd.Timedelta(days=1), periods=7)
    prediction_df = pd.DataFrame({"Date": future_dates, "Predicted_Actual_Units": future_predictions.astype(int)})
    st.dataframe(prediction_df)
else:
    st.info("Not enough historical data for prediction — need at least 7 days.")

st.markdown("---")

# ⚙️ Bottleneck Analysis Based on Scenario
st.subheader("⚙️ Bottleneck Analysis Based on Scenario")
total_simulated_downtime_impact = sim_downtime * len(simulated_df)
total_simulated_material_shortage_impact = sim_material_shortage * len(simulated_df)

if total_simulated_downtime_impact > total_simulated_material_shortage_impact:
    st.error(f"🧯 Bottleneck Detected: Downtime is the dominant factor. Total simulated downtime impact: {total_simulated_downtime_impact} minutes.")
elif total_simulated_material_shortage_impact > total_simulated_downtime_impact:
    st.error(f"📦 Bottleneck Detected: Material Shortage is the dominant factor. Total simulated material shortfall impact: {total_simulated_material_shortage_impact:.2f}%.")
else:
    st.success("✅ Balanced Scenario: Downtime and Material Shortage are contributing equally.")

st.markdown("---")

# 🎨 Custom styling for visual appeal
st.markdown("""
    <style>
    .stApp {
        background-color: #f4f6f8;
    }
    .st-eb {
        font-size: 18px !important;
        color: #1f77b4;
    }
    h1, h2 {
        color: #003366;
    }
    </style>
""", unsafe_allow_html=True)

st.caption("Developed by Nikhil Sharma — Production Efficiency Simulator")
