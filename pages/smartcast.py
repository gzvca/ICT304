# remember to import all the needed libs in the terminal
# pip install lightgbm pandas streamlit numpy plotly scikit-learn

import streamlit as st
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor


CATEGORY_MAP = {
    "Category_003": "Dairy_Products",
    "Category_008": "Canned_Goods",
    "Category_010": "Instant_Foods",
    "Category_012": "Meat_Seafood",
    "Category_013": "Fresh_Produce",
    "Category_017": "Bakery_Items",
    "Category_022": "Beverages",
    "Category_026": "Snacks",
    "Category_029": "Frozen_Foods",
    "Category_031": "Household_Goods",
}

KEEP_CATEGORIES = list(CATEGORY_MAP.keys())
FEATURE_COLS = [
    "Category_enc", "rolling_7d_avg", "rolling_30d_avg",
    "demand_lag_7", "demand_lag_14", "demand_lag_30",
    "pct_change_7d", "pct_change_30d",
    "month", "day", "dayofweek", "weekofyear",
    "SchoolHoliday", "state_holiday_encoded",
]
TARGET_COL    = "Order_Demand"
FORECAST_DAYS = 14
SPLIT_DATE    = pd.Timestamp("2016-10-01")
 

def inject_css():
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(180deg, #f4f8fb 0%, #eef4f9 100%);
        }

        .block-container {
            max-width: 1200px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        .hero-wrap {
            background: linear-gradient(135deg, #0B2A4A 0%, #1e4f82 50%, #2F6FA3 100%);
            border-radius: 28px;
            padding: 34px 36px;
            color: white;
            margin-bottom: 22px;
            box-shadow: 0 16px 35px rgba(11, 42, 74, 0.18);
            position: relative;
            overflow: hidden;
        }

        .hero-wrap::after {
            content: "";
            position: absolute;
            right: -60px;
            top: -60px;
            width: 220px;
            height: 220px;
            background: rgba(255,255,255,0.08);
            border-radius: 50%;
        }

        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
        }

        .hero-subtitle {
            font-size: 1.06rem;
            line-height: 1.7;
            opacity: 0.95;
            max-width: 780px;
            position: relative;
            z-index: 1;
        }

        .panel {
            background: rgba(255,255,255,0.92);
            border: 1px solid #d7e4ef;
            border-radius: 24px;
            padding: 22px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 18px;
        }

        .panel-title {
            font-size: 1.22rem;
            font-weight: 800;
            color: #0B2A4A;
            margin-bottom: 12px;
        }
        
        .stat-row { 
            display:flex; 
            gap:20px;
            flex-wrap: wrap;
            margin-bottom:20px;
            justify-content: flex-start;
        }

        .stat-card {
            border-radius: 16px;
            padding: 18px;
            color: white;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
            margin-bottom: 10px;
            min-width: 180px;
            flex: 1;
        }
        
        .stat-blue {
            background: linear-gradient(135deg, #2F6FA3 0%, #4f8fc4 100%);
        }

        .stat-dark {
            background: linear-gradient(135deg, #0B2A4A 0%, #234e7b 100%);
        }

        .stat-label {
            font-size: 0.92rem;
            opacity: 0.92;
            margin-bottom: 8px;
        }

        .stat-value {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.05;
        }

        .stButton > button {
            width: 200px;
            height: 50px;
            background: linear-gradient(135deg, #2F6FA3 0%, #3f82bb 100%);
            color: white;
            border-radius: 14px;
            font-weight: 800;
            border: none;
            padding: 0.78rem 1rem;
            box-shadow: 0 8px 18px rgba(47, 111, 163, 0.22);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #0B2A4A 0%, #1e4f82 100%);
            color: white;
        }

        .stButton > button:focus:not(:active) {
            color: white;
            border-color: transparent;
        }
        
        .note {
            color: #64748B;
            font-size: 0.94rem;
            margin-top: 8px;
            line-height: 1.6;
        }
        
        .stSelectbox label { font-weight:700; color:#0B2A4A; }
    
        
        </style>
        """,
        unsafe_allow_html=True
    )

def render_header():
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-title">SmartCast</div>
            <div class="hero-subtitle">
                AI-powered Future Demand Forecasting 
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes: bytes):
 
    # 1. Load Data from user input
    df = pd.read_csv(io.BytesIO(file_bytes), sep=";")
    df.drop(
        [c for c in ["Product_Code","Product_id","Open","Petrol_price","Warehouse","Promo"] if c in df.columns],
        axis=1, inplace=True, errors="ignore"
    )
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
 
    df_merged = (
        df.groupby(["Date","Product_Category"])
        .agg(Order_Demand=("Order_Demand","sum"),
            StateHoliday=("StateHoliday","max"),
            SchoolHoliday=("SchoolHoliday","max"))
        .reset_index()
    )
 
    # 2. Filter & rename categories 
    df_cleaned = df_merged[df_merged["Product_Category"].isin(KEEP_CATEGORIES)].copy()
    df_cleaned["Product_Category"] = df_cleaned["Product_Category"].map(CATEGORY_MAP)
 
    # 3. Daily aggregation 
    daily = (
        df_cleaned.groupby(["Product_Category","Date"])
                .agg(Order_Demand=("Order_Demand","sum"),
                    StateHoliday=("StateHoliday","max"),
                    SchoolHoliday=("SchoolHoliday","max"))
                .reset_index()
                .sort_values(["Product_Category","Date"])
                .reset_index(drop=True)
    )
 
    # 4. Feature engineering 
    results = []
    for cat, grp in daily.groupby("Product_Category"):
        grp = grp.sort_values("Date").copy()
        grp["day_of_week"]     = grp["Date"].dt.day_name()
        grp["month"]           = grp["Date"].dt.month_name()
        grp["rolling_7d_avg"]  = grp["Order_Demand"].rolling(7,  min_periods=1).mean()
        grp["rolling_30d_avg"] = grp["Order_Demand"].rolling(30, min_periods=1).mean()
        grp["demand_lag_7"]    = grp["Order_Demand"].shift(7)
        grp["demand_lag_14"]   = grp["Order_Demand"].shift(14)
        grp["demand_lag_30"]   = grp["Order_Demand"].shift(30)
        grp["pct_change_7d"]   = grp["Order_Demand"].pct_change(7)  * 100
        grp["pct_change_30d"]  = grp["Order_Demand"].pct_change(30) * 100
        results.append(grp)
    featured = pd.concat(results, ignore_index=True)
 
    # 5. Encoding & cleaning
    state_mapping = {"0":0, "a":1, "b":2}
    featured["state_holiday_encoded"] = featured["StateHoliday"].map(state_mapping).fillna(0).astype(int)
    featured.drop(columns=["StateHoliday"], inplace=True, errors="ignore")
 
    le = LabelEncoder()
    featured["Category_enc"] = le.fit_transform(featured["Product_Category"])
 
    featured["year"]      = featured["Date"].dt.year
    featured["month"]     = featured["Date"].dt.month
    featured["day"]       = featured["Date"].dt.day
    featured["dayofweek"] = featured["Date"].dt.dayofweek
    featured["weekofyear"]= featured["Date"].dt.isocalendar().week.astype(int)
 
    lag_cols = ["demand_lag_7","demand_lag_14","demand_lag_30","pct_change_7d","pct_change_30d"]
    featured[lag_cols] = featured[lag_cols].fillna(0)
    featured = featured.replace([np.inf, -np.inf], np.nan).fillna(0)
 
    # 6. Train / test split
    train_df = featured[featured["Date"] < SPLIT_DATE].copy()
    test_df  = featured[featured["Date"] >= SPLIT_DATE].copy()
    X_train  = train_df[FEATURE_COLS]
    y_train  = train_df[TARGET_COL]
    X_test   = test_df[FEATURE_COLS]
    y_test   = test_df[TARGET_COL]
 
    # 7. Train LightGBM (No grid search as it will take along time to run. Instead, the optimized parameters are already chosen beforehand)
    lgb_model = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        num_leaves=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0, reg_lambda=1, min_child_samples=10,
        n_jobs=max(1, multiprocessing.cpu_count() // 2),
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
 
    def smape(a, p):
        a, p = np.array(a), np.array(p)
        d = (np.abs(a) + np.abs(p)) / 2
        m = d != 0
        return float(np.mean(np.abs(a[m]-p[m]) / d[m]) * 100)
 
    metrics = {
        "MAE":   round(float(mean_absolute_error(y_test, lgb_pred)), 2),
        "RMSE":  round(float(np.sqrt(mean_squared_error(y_test, lgb_pred))), 2),
        "SMAPE": round(smape(y_test, lgb_pred), 2),
    }
 
    # 8. 14-day forecast 
    last_date    = featured["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS)
 
    school_holiday_avg = (
        featured.groupby(featured["Date"].dt.dayofyear)["SchoolHoliday"]
                .mean().round().astype(int).to_dict()
    )
    state_holiday_avg = (
        featured.groupby(featured["Date"].dt.dayofyear)["state_holiday_encoded"]
                .mean().round().astype(int).to_dict()
    )
 
    model_features = X_train.columns.tolist()
    all_forecasts  = []
 
    for category in featured["Product_Category"].unique():
        cat_data    = featured[featured["Product_Category"] == category].sort_values("Date")
        hist_cap    = cat_data["Order_Demand"].quantile(0.95) * 1.2
        demand_hist = list(cat_data["Order_Demand"].values.astype(float))
        daily_preds = []
 
        for date in future_dates:
            row = {
                "month":               date.month,
                "day":                 date.day,
                "dayofweek":           date.dayofweek,
                "weekofyear":          date.isocalendar()[1],
                "Category_enc":        cat_data["Category_enc"].iloc[0],
                "SchoolHoliday":       school_holiday_avg.get(date.timetuple().tm_yday, 0),
                "state_holiday_encoded": state_holiday_avg.get(date.timetuple().tm_yday, 0),
                "rolling_7d_avg":      np.mean(demand_hist[-7:]),
                "rolling_30d_avg":     np.mean(demand_hist[-30:]),
                "demand_lag_7":        demand_hist[-7],
                "demand_lag_14":       demand_hist[-14],
                "demand_lag_30":       demand_hist[-30],
                "pct_change_7d":  (demand_hist[-1]-demand_hist[-7])  / (demand_hist[-7]  + 1e-5),
                "pct_change_30d": (demand_hist[-1]-demand_hist[-30]) / (demand_hist[-30] + 1e-5),
            }
            X_row = pd.DataFrame([row])[model_features]
            pred  = float(lgb_model.predict(X_row)[0])
            pred  = min(max(pred, 0), hist_cap)
            demand_hist.append(pred)
            daily_preds.append(pred)
 
        all_forecasts.append(pd.DataFrame({
            "Date":             future_dates,
            "Product_Category": category,
            "Predicted_Demand": np.maximum(daily_preds, 0).round().astype(int),
        }))
 
    final_forecast = pd.concat(all_forecasts).reset_index(drop=True)
 
    return featured, final_forecast, metrics, last_date
 
 
#  PLOTLY CHART
def build_forecast_chart(featured: pd.DataFrame, final_forecast: pd.DataFrame,
                        category: str, last_date) -> go.Figure:
    hist = (featured[featured["Product_Category"] == category]
                    .sort_values("Date").tail(30))
    fc   = final_forecast[final_forecast["Product_Category"] == category]

    last_row     = hist.iloc[-1]
    bridge_dates = [last_row["Date"]] + list(fc["Date"])
    bridge_vals  = [last_row["Order_Demand"]] + list(fc["Predicted_Demand"])

    fig = go.Figure()

    # Historical area
    fig.add_trace(go.Scatter(
        x=hist["Date"], y=hist["Order_Demand"],
        mode="lines", name="Historical Demand",
        line=dict(color="#2F6FA3", width=2),
        fill="tozeroy", fillcolor="rgba(47,111,163,0.08)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: %{y:,}<extra></extra>",
    ))

    # Bridge connector
    fig.add_trace(go.Scatter(
        x=[bridge_dates[0], bridge_dates[1]],
        y=[bridge_vals[0],  bridge_vals[1]],
        mode="lines", showlegend=False,
        line=dict(color="#f97316", width=2, dash="dot"),
        hoverinfo="skip",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc["Date"], y=fc["Predicted_Demand"],
        mode="lines+markers", name="14-Day Forecast",
        line=dict(color="#f97316", width=2.5, dash="dash"),
        marker=dict(color="#f97316", size=7, symbol="circle",
                    line=dict(color="white", width=1.5)),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.07)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast: %{y:,}<extra></extra>",
    ))

    # Vertical rule
    fig.add_vline(
        x=pd.Timestamp(last_date).timestamp() * 1000, 
        line_width=1.5, line_dash="dot", line_color="#94a3b8",
        annotation_text="Forecast →",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#64748b"),
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Syne, sans-serif", color="#1e293b"),
        margin=dict(t=20, r=20, b=50, l=60),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor="#d7e4ef",
                        font=dict(family="Space Mono, monospace", size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0,
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0", linecolor="#e2e8f0",
                tickformat="%d %b", tickfont=dict(size=11, color="#64748b")),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0", linecolor="#e2e8f0",
                title="Order Demand", tickfont=dict(size=11, color="#64748b"),
                tickformat=",", rangemode="tozero"),
        height=420,
    )
    return fig
    
def render_stats(featured, final_forecast, category, metrics):
    fc_cat   = final_forecast[final_forecast["Product_Category"] == category]
    hist_cat = featured[featured["Product_Category"] == category]
 
    total_fc   = int(fc_cat["Predicted_Demand"].sum())
    avg_hist   = int(hist_cat["Order_Demand"].mean())
    peak_fc    = int(fc_cat["Predicted_Demand"].max())
    last_act   = int(hist_cat.sort_values("Date")["Order_Demand"].iloc[-1])
    day1_fc    = int(fc_cat.sort_values("Date")["Predicted_Demand"].iloc[0])
    chg        = round((day1_fc - last_act) / (last_act + 1e-5) * 100, 1)
    sign       = "+" if chg >= 0 else ""
    chg_color  = "green" if chg >= 0 else "orange"
 
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card stat-dark">
            <div class="stat-label">14-Day Total Forecast</div>
            <div class="stat-value">{total_fc:,}</div>
        </div>
        <div class="stat-card stat-blue">
            <div class="stat-label">Avg Historical Demand</div>
            <div class="stat-value">{avg_hist:,}</div>
        </div>
        <div class="stat-card stat-dark">
            <div class="stat-label">Forecast Peak</div>
            <div class="stat-value">{peak_fc:,}</div>
        </div>
        <div class="stat-card stat-blue">
            <div class="stat-label">vs Last Actual</div>
            <div class="stat-value {chg_color}">{sign}{chg}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)    
  
    
def render(go_to):
    inject_css()
    render_header()

        
    top_cols = st.columns([1, 4])
    with top_cols[0]:
        if st.button("← Back", use_container_width=True):
            go_to("home")
            st.rerun()
            
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Upload a CSV file</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note">Due to limitations, please only upload the given "Retail.csv" file. </div>',
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader(
        "Upload your CSV",
        type=["csv"],
        label_visibility="collapsed",
        accept_multiple_files= False,
        help="Only CSV files allowed",
        key="smartcast_upload"
        )
        
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is None:
        st.warning("⚠️ No file uploaded. Please upload Retail.csv to continue.")
        return

    if uploaded_file.name != "Retail.csv":
        st.error("Incorrect file. Please upload **Retail.csv** only.")
        return
            
    try:
        preview = pd.read_csv(uploaded_file, sep=";", nrows=5)
        uploaded_file.seek(0)  # reset for pipeline
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(preview, use_container_width=True)
    st.success("✅ CSV file successfully uploaded!")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Predict Demand for 14 Days</div>', unsafe_allow_html=True)
 
    if st.button("  Run Forecast"):
        with st.spinner("Training LightGBM model and generating forecast…"):
            file_bytes = uploaded_file.read()
            featured, final_forecast, metrics, last_date = run_pipeline(file_bytes)
            st.session_state["forecast_ready"]  = True
            st.session_state["featured"]        = featured
            st.session_state["final_forecast"]  = final_forecast
            st.session_state["metrics"]         = metrics
            st.session_state["last_date"]       = last_date
        st.success("✅ Forecast complete!")
 
    st.markdown("</div>", unsafe_allow_html=True)
 
    # Forecast results
    if st.session_state.get("forecast_ready"):
        featured       = st.session_state["featured"]
        final_forecast = st.session_state["final_forecast"]
        metrics        = st.session_state["metrics"]
        last_date      = st.session_state["last_date"]
 
        categories = sorted(final_forecast["Product_Category"].unique().tolist())
 
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📊 14-Day Demand Forecast</div>', unsafe_allow_html=True)
 
        selected_cat = st.selectbox(
            "Select Product Category",
            options=categories,
            index=0,
            key="cat_select"
        )
 
        # Stats
        render_stats(featured, final_forecast, selected_cat, metrics)
 
        # Chart
        fig = build_forecast_chart(featured, final_forecast, selected_cat, last_date)
        st.plotly_chart(fig, use_container_width=True, config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
        })
 
        # Forecast table
        with st.expander(" View detailed forecast data table"):
            fc_table = (
                final_forecast[final_forecast["Product_Category"] == selected_cat]
                .sort_values("Date")
                .reset_index(drop=True)
            )
            fc_table["Date"] = fc_table["Date"].dt.strftime("%d %b %Y")
            st.dataframe(fc_table[["Date","Product_Category","Predicted_Demand"]],
                        use_container_width=True)
 
        st.markdown("</div>", unsafe_allow_html=True)
 
        # Model metrics footer
        st.markdown(
            f'<div class="note" style="text-align:center;padding-top:4px;">'
            f'Model performance on held-out test set — '
            f'MAE: <strong>{metrics["MAE"]:,}</strong> &nbsp;|&nbsp; '
            f'RMSE: <strong>{metrics["RMSE"]:,}</strong> &nbsp;|&nbsp; '
            f'SMAPE: <strong>{metrics["SMAPE"]}%</strong>'
            f'</div>',
            unsafe_allow_html=True
        )
    
