# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import kagglehub
from io import BytesIO
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(page_title="Advanced Fraud Detection Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <div style="background: linear-gradient(90deg, #1E88E5, #42A5F5); padding: 20px; border-radius: 10px; text-align: center;">
        <h1 style="color: white; margin: 0;">Marina Auditing Visualization Tool (MAVT)</h1>
        <h3 style="color: #E3F2FD; margin: 5px 0 0;">Data Visualization Tools and Techniques in Business Informatics</h3>
    </div>
""", unsafe_allow_html=True)

# Load dataset with validation
@st.cache_data
def load_data(uploaded_file=None):
    with st.spinner("Loading dataset..."):
        try:
            if uploaded_file is None:
                path = kagglehub.dataset_download("saigeethasb/credit-card-fraud")
                df = pd.read_csv(path + "/credit_card_fraud_synthetic.csv")
            else:
                df = pd.read_csv(uploaded_file)
            df.dropna(inplace=True)
            if 'Timestamp' not in df.columns:
                df['Timestamp'] = [datetime.now() - timedelta(days=i % 90) for i in range(len(df))]
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            if df.duplicated(subset=['Transaction_ID']).sum() > 0:
                st.warning(f"Removed {df.duplicated(subset=['Transaction_ID']).sum()} duplicate transactions.")
                df = df.drop_duplicates(subset=['Transaction_ID'])
            if df['Transaction_Amount'].min() < 0:
                st.error("Negative transaction amounts detected. Corrected to absolute values.")
                df['Transaction_Amount'] = df['Transaction_Amount'].abs()
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            logger.error(f"Data loading error: {str(e)}")
            return None

# Initialize session state
for key in ['df', 'model_trained', 'results', 'annotations', 'audit_log', 'feature_importance', 'scaler', 'best_model', 'X', 'X_test_scaled', 'y_test']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'model_trained' else False
        st.session_state[key] = {} if key == 'annotations' else st.session_state[key]
        st.session_state[key] = [] if key == 'audit_log' else st.session_state[key]

# Sidebar for configuration
with st.sidebar:
    st.header("Control Panel")
    st.markdown("<h4 style='color: #1E88E5;'>Data & Settings</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Custom CSV Dataset", type=["csv"], help="Upload your dataset or use default Kaggle data.")
    if st.button("Load Default Kaggle Dataset", key="load_default"):
        df = load_data()
        if df is not None:
            st.session_state.df = df
            st.session_state.model_trained = False
            st.success(f"Loaded default dataset: {df.shape[0]} transactions")
    threshold = st.slider("Fraud Probability Threshold", 0.1, 0.9, 0.7, 0.05, help="Set the cutoff for fraud detection.")
    compliance_limit = st.number_input("Compliance Amount Limit ($)", min_value=0.0, value=1000.0, step=100.0, help="Flag transactions above this amount.")
   
# Main logic
if st.session_state.df is not None:
    df = st.session_state.df

    # Train models
    def train_models(df, threshold, severity_weight, compliance_limit):
        st.write("Training fraud detection models...")
        try:
            X = df.drop(['Is_Fraudulent', 'Transaction_ID', 'Timestamp', 'Location', 'Transaction_Type'], axis=1)
            y = df['Is_Fraudulent']
            st.session_state.X = X
            smote = SMOTE(random_state=42)
            X_smote, y_smote = smote.fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.session_state.scaler = scaler
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.y_test = y_test

            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'XGBoost': XGBClassifier(eval_metric='logloss', n_jobs=-1),
                'Naive Bayes': GaussianNB(),
                'Isolation Forest': IsolationForest(random_state=42, contamination=0.1),
                'KMeans': KMeans(n_clusters=2, random_state=42),
                'MLP': MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
            }

            results = {}
            for name, model in models.items():
                if name in ['Isolation Forest', 'KMeans']:
                    model.fit(X_train_scaled)
                    predictions = model.predict(X_test_scaled)
                    predictions = [1 if p == -1 else 0 for p in predictions] if name == 'Isolation Forest' else [1 if p else 0 for p in predictions]
                    results[name] = {
                        'f1': f1_score(y_test, predictions),
                        'accuracy': accuracy_score(y_test, predictions),
                        'precision': precision_score(y_test, predictions),
                        'recall': recall_score(y_test, predictions),
                        'roc_auc': 0,
                        'model': model,
                        'predictions': predictions,
                        'probs': None
                    }
                else:
                    model.fit(X_train_scaled, y_train)
                    predictions = model.predict(X_test_scaled)
                    probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else predictions
                    results[name] = {
                        'f1': f1_score(y_test, predictions),
                        'accuracy': accuracy_score(y_test, predictions),
                        'precision': precision_score(y_test, predictions),
                        'recall': recall_score(y_test, predictions),
                        'roc_auc': roc_auc_score(y_test, probs) if hasattr(model, 'predict_proba') else 0,
                        'model': model,
                        'predictions': predictions,
                        'probs': probs
                    }
                if name == 'Random Forest':
                    st.session_state.feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)

            best_model_name = max(results, key=lambda x: results[x]['f1'])
            st.session_state.best_model = results[best_model_name]['model']
            X_scaled = scaler.transform(X)
            best_probs = st.session_state.best_model.predict_proba(X_scaled)[:, 1] if hasattr(st.session_state.best_model, 'predict_proba') else st.session_state.best_model.predict(X_scaled)
            df['Fraud_Probability'] = best_probs
            df['Predicted'] = (df['Fraud_Probability'] >= threshold).astype(int)
            df['Severity_Score'] = df['Fraud_Probability'] * df['Transaction_Amount'] * severity_weight
            df['Compliance_Flag'] = (df['Transaction_Amount'] > compliance_limit).astype(int)
            df['Risk_Category'] = pd.cut(df['Fraud_Probability'], bins=[0, 0.3, 0.7, 1], labels=['Low', 'Medium', 'High'], include_lowest=True)

            st.success(f"Analysis Complete! Best Model: {best_model_name} (F1: {results[best_model_name]['f1']:.4f})")
            logger.info(f"Models trained. Best model: {best_model_name}")
            return df, results
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            logger.error(f"Training failed: {str(e)}")
            return df, None

    # Update derived columns
    def update_derived_columns(df, threshold, severity_weight, compliance_limit):
        if st.session_state.best_model and st.session_state.X is not None:
            X_scaled = st.session_state.scaler.transform(st.session_state.X)
            best_probs = st.session_state.best_model.predict_proba(X_scaled)[:, 1] if hasattr(st.session_state.best_model, 'predict_proba') else st.session_state.best_model.predict(X_scaled)
            df['Fraud_Probability'] = best_probs
            df['Predicted'] = (df['Fraud_Probability'] >= threshold).astype(int)
            df['Severity_Score'] = df['Fraud_Probability'] * df['Transaction_Amount'] * severity_weight
            df['Compliance_Flag'] = (df['Transaction_Amount'] > compliance_limit).astype(int)
            df['Risk_Category'] = pd.cut(df['Fraud_Probability'], bins=[0, 0.3, 0.7, 1], labels=['Low', 'Medium', 'High'], include_lowest=True)
            st.session_state.df = df
            st.success(f"Dashboard refreshed with threshold {threshold}.")
            logger.info(f"Derived columns updated with threshold {threshold}")

    # Engineering Section
    st.markdown("<h2 style='color: #1E88E5;'>Engineering Insights</h2>", unsafe_allow_html=True)
    if not st.session_state.model_trained:
        st.info("Run the analysis to see engineering insights.")
        if st.button("Run Fraud Detection Analysis", key="run_analysis"):
            with st.spinner("Running fraud detection..."):
                df, results = train_models(df, threshold, severity_weight, compliance_limit)
                st.session_state.df = df
                st.session_state.model_trained = True
                st.session_state.results = results
                st.session_state.audit_log.append({'Action': 'Analysis Run', 'Timestamp': datetime.now()})
    else:
        if st.sidebar.button("Refresh Analysis", key="refresh_analysis"):
            with st.spinner("Updating dashboard..."):
                update_derived_columns(df, threshold, severity_weight, compliance_limit)

    if st.session_state.model_trained and st.session_state.results is not None:
        df = st.session_state.df
        results = st.session_state.results

        st.subheader("Pre-ML Data Look")
        col1, col2 = st.columns(2)
        with col1:
            fig_pre = px.scatter(df, x='Transaction_Amount', y='Deviation_From_Avg', color='Is_Fraudulent',
                                 title='Amount vs Deviation', labels={'Transaction_Amount': 'Amount ($)', 'Deviation_From_Avg': 'Deviation'})
            st.plotly_chart(fig_pre)
        with col2:
            fig_box = px.box(df, x='Is_Fraudulent', y='Transaction_Amount', title='Amount Spread by Fraud')
            st.plotly_chart(fig_box)

        st.subheader("Model Performance")
        comparison_df = pd.DataFrame([(k, v['f1'], v['accuracy'], v['precision'], v['recall'], v['roc_auc'])
                                      for k, v in results.items()],
                                     columns=['Model', 'F1 Score', 'Accuracy', 'Precision', 'Recall', 'ROC-AUC'])
        fig = go.Figure()
        for col in ['F1 Score', 'Accuracy', 'Precision', 'Recall', 'ROC-AUC']:
            fig.add_trace(go.Bar(x=comparison_df['Model'], y=comparison_df[col], name=col))
        fig.update_layout(title='Model Comparison', barmode='group', yaxis_title='Score', xaxis_title='Model',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig)
        best_model_name = max(results, key=lambda x: results[x]['f1'])
        st.markdown(f"<p style='font-size: 18px; color: #388E3C;'><b>Best Model: {best_model_name}</b> (F1 Score: {results[best_model_name]['f1']:.4f})</p>", unsafe_allow_html=True)

        st.subheader(f"Details: {best_model_name}")
        col3, col4 = st.columns(2)
        with col3:
            if results[best_model_name]['probs'] is not None:
                fpr, tpr, _ = roc_curve(st.session_state.y_test, results[best_model_name]['probs'])
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC = {results[best_model_name]['roc_auc']:.2f}", line=dict(color='#1E88E5')))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='grey'), name='Random'))
                fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc)
        with col4:
            cm = confusion_matrix(st.session_state.y_test, results[best_model_name]['predictions'])
            fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['No Fraud', 'Fraud'], y=['No Fraud', 'Fraud'], text=cm, texttemplate='%{text}', colorscale='Blues'))
            fig_cm.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig_cm)

        if st.session_state.feature_importance is not None:
            st.subheader("Key Features")
            fig_fi = px.bar(st.session_state.feature_importance, x='Feature', y='Importance', title="Feature Importance (Random Forest)",
                            color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig_fi)

    # Auditing Section
    st.markdown("<h2 style='color: #1E88E5;'>Auditing Dashboard</h2>", unsafe_allow_html=True)
    if st.session_state.model_trained:
        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Investigation", "Visuals", "Reports"])

        with tab1:  # Summary
            st.subheader("Audit Overview")
            cols = st.columns(5)
            metrics = {
                "Total Transactions": len(df),
                "Flagged Fraud": df['Predicted'].sum(),
                "Actual Fraud": df['Is_Fraudulent'].sum(),
                "High Risk": len(df[df['Risk_Category'] == 'High']),
                "Compliance Issues": df['Compliance_Flag'].sum()
            }
            colors = ['#1E88E5', '#FF5722', '#FFD700', '#4CAF50', '#AB47BC']
            for i, (label, value) in enumerate(metrics.items()):
                with cols[i]:
                    st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, {colors[i]}, {colors[i]}BB);">
                            <p class="metric-label">{label}</p>
                            <p class="metric-value">{value:,}</p>
                        </div>
                    """, unsafe_allow_html=True)

            st.subheader("Risk Spread")
            fig_pie = px.pie(df['Risk_Category'].value_counts().reset_index(), names='Risk_Category', values='count',
                             title='Risk Distribution', color_discrete_map={'Low': '#87CEFA', 'Medium': '#00CED1', 'High': '#006400'})
            st.plotly_chart(fig_pie)

        with tab2:  # Investigation
            st.subheader("Transaction Check")
            risk_filter = st.multiselect("Filter by Risk", options=['Low', 'Medium', 'High'], default=['High'])
            flagged_only = st.checkbox("Show Flagged Only", value=True)
            filtered_df = df[df['Risk_Category'].isin(risk_filter) & (df['Predicted'] == 1 if flagged_only else True)]
            st.write("Top 10 High-Severity Transactions")
            st.dataframe(filtered_df.nlargest(10, 'Severity_Score')[['Transaction_ID', 'Transaction_Amount', 'Fraud_Probability', 'Severity_Score', 'Risk_Category']]
                         .style.format({'Fraud_Probability': "{:.2%}", 'Severity_Score': "{:.2f}"}))

            st.subheader("Audit Notes")
            if not filtered_df.empty:
                transaction_id = st.selectbox("Select Transaction ID", filtered_df['Transaction_ID'].unique())
                note = st.text_area("Add/Edit Note", value=st.session_state.annotations.get(str(transaction_id), {}).get('note', ""))
                if st.button("Save Note", key="save_note"):
                    st.session_state.annotations[str(transaction_id)] = {'note': note, 'timestamp': datetime.now()}
                    st.session_state.audit_log.append({'Action': 'Note Added', 'Transaction_ID': transaction_id, 'Timestamp': datetime.now()})
                    st.success("Note saved!")
                    logger.info(f"Note saved for Transaction_ID: {transaction_id}")

            st.subheader("Audit Trail")
            if st.session_state.audit_log:
                st.dataframe(pd.DataFrame(st.session_state.audit_log).style.format({'Timestamp': '{:%Y-%m-%d %H:%M:%S}'}))

        with tab3:  # Visuals
            st.subheader("Interactive Charts")
            start_date = st.date_input("Start Date", df['Timestamp'].min().date(), key="start_date")
            end_date = st.date_input("End Date", df['Timestamp'].max().date(), key="end_date")
            filtered_df = df[(df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)]

            row1 = st.columns(2)
            with row1[0]:
                fig_hist = px.histogram(filtered_df, x='Fraud_Probability', nbins=50, title='Fraud Probability Spread',
                                        labels={'Fraud_Probability': 'Probability'})
                fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
                st.plotly_chart(fig_hist)
            with row1[1]:
                fig_trend = px.line(filtered_df.groupby(filtered_df['Timestamp'].dt.date).agg({'Predicted': 'sum'}),
                                    title='Fraud Trend Over Time', labels={'value': 'Fraud Count', 'Timestamp': 'Date'})
                st.plotly_chart(fig_trend)

            row2 = st.columns(2)
            with row2[0]:
                fig_heatmap = px.density_heatmap(filtered_df, x='Transaction_Amount', y='Fraud_Probability',
                                                 title='Heatmap: Amount vs Fraud Probability')
                st.plotly_chart(fig_heatmap)
            with row2[1]:
                fig_stacked = px.bar(filtered_df.groupby('Transaction_Type').agg({'Predicted': 'sum', 'Is_Fraudulent': 'sum'}).reset_index(),
                                     x='Transaction_Type', y=['Predicted', 'Is_Fraudulent'], barmode='stack',
                                     title='Fraud by Transaction Type', labels={'value': 'Count', 'variable': 'Category'})
                st.plotly_chart(fig_stacked)

            fig_donut = go.Figure(data=[go.Pie(labels=filtered_df['Risk_Category'].value_counts().index,
                                               values=filtered_df['Risk_Category'].value_counts().values, hole=0.4,
                                               marker=dict(colors=['#87CEFA', '#00CED1', '#006400']))])
            fig_donut.update_layout(title="Risk Spread (Donut)")
            st.plotly_chart(fig_donut)

        with tab4:  # Reports
            st.subheader("Audit Report")
            report_text = f"""Credit Card Fraud Detection Audit Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset Size: {len(df):,} transactions
Flagged Fraud: {df['Predicted'].sum():,}
Actual Fraud: {df['Is_Fraudulent'].sum():,}
High-Risk Transactions: {len(df[df['Risk_Category'] == 'High']):,}
Compliance Violations: {df['Compliance_Flag'].sum():,}
Best Model: {best_model_name} (F1 Score: {results[best_model_name]['f1']:.4f})
Top Features: {', '.join(st.session_state.feature_importance['Feature'].head(3).tolist()) if st.session_state.feature_importance is not None else 'N/A'}
Audit Notes: {', '.join([f"{tid}: {data['note']}" for tid, data in st.session_state.annotations.items()]) if st.session_state.annotations else 'None recorded'}
Recommendations: Focus on high-severity transactions for review.
"""
            st.text_area("Report Preview", report_text, height=300)
            st.download_button("Download Report", report_text, "audit_report.txt", "text/plain", key="download_report")
            csv = df.to_csv(index=False)
            st.download_button("Download Results", data=csv, file_name="fraud_detection_results.csv", mime="text/csv", key="download_csv")

# Custom CSS for a clean look
st.markdown("""
    <style>
    .stApp {
        background-color: #F4F6F9;
        font-family: 'Segoe UI', Arial, sans-serif;
        padding: 20px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1E88E5, #42A5F5);
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1565C0, #1976D2);
        color: white;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E88E5;
        color: white;
        padding: 12px 24px;
        border-radius: 8px 8px 0 0;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #1565C0;
    }
    .metric-card {
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        color: white;
    }
    .metric-label {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    h2 {
        color: #1E88E5;
        border-bottom: 2px solid #BBDEFB;
        padding-bottom: 5px;
    }
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] div:not(.metric-card),
    .stApp [data-testid="stText"] {
        color: #000000;
    }
    [data-testid="stSidebar"] {
        background-color: #2E2E2E;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stButton>button {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Footer
st.markdown("<hr><p style='text-align: center; color: #666;'>Developed by Marina Maged Rasmy | Graduation Project 2025</p>", unsafe_allow_html=True)

# Note: Save this as 'app.py' and follow deployment steps below for Streamlit Community Cloud
