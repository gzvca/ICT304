import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#import numpy as np
#import plotly.express as px
#import plotly.graph_objects as go
#import lightgbm as lgb
#import multiprocessing
#import warnings
#warnings.filterwarnings("ignore")
#from plotly.subplots import make_subplots
#from ipywidgets import widgets

#from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.metrics import mean_absolute_error, mean_squared_error
#from sklearn.preprocessing import LabelEncoder,  MinMaxScaler
#from sklearn.model_selection import GridSearchCV
#from lightgbm import LGBMRegressor

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

        .stat-card {
            border-radius: 20px;
            padding: 18px;
            color: white;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
            margin-bottom: 10px;
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
    
    uploaded_file = st.file_uploader(
        "Upload your CSV",
        type=["csv"],
        label_visibility="collapsed",
        accept_multiple_files= False,
        help="Only CSV files allowed"
        )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(df.head())
            st.success("CSV file successfully uploaded!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.error("⚠️ No file uploaded. Please upload a CSV file to continue.")