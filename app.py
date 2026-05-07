# streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import base64

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="Factor Analysis - Heart Disease",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card style for metrics */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Title styling */
    .main-title {
    background: linear-gradient(135deg, #FF1493 0%, #FF69B4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em;
    font-weight: bold;
    text-align: center;
    margin-bottom: 30px;

    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255,255,255,0.95);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 30px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 **Factor Analysis Settings**")
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader("📁 Upload CSV File", type=['csv'])
    
    # Analysis parameters
    st.markdown("### ⚙️ **Parameters**")
    
    n_factors = st.slider(
        "Number of Factors", 
        min_value=2, 
        max_value=10, 
        value=5,
        help="Select the number of factors to extract"
    )
    
    rotation = st.selectbox(
    "Rotation Method",
    ['varimax', 'oblimin', 'promax', 'quartimax', 'equamax', None],  # ✅ None بدلاً من 'none'
    help="Rotation helps interpret factors more easily"
    
  )
    
    
    method = st.selectbox(
        "Extraction Method",
        ['ml', 'minres', 'principal'],
        help="Method for factor extraction"
    )
    
    # Color theme
    color_theme = st.color_picker("🎨 Pick Theme Color", "#FF69B4")
    
    st.markdown("---")
    st.markdown("### 📊 **Dataset Info**")
    st.markdown("- 303 patients")
    st.markdown("- 13 medical features")
    st.markdown("- Target: Heart Disease")

# Main content
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Main title
    st.markdown('<div class="main-title">❤️ Factor Analysis - Heart Disease Dataset</div>', unsafe_allow_html=True)
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Samples</h3>
            <h2>{df.shape[0]}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔢 Features</h3>
            <h2>{df.shape[1]}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>❤️ Disease Present</h3>
            <h2>{df['target'].sum()}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💚 Healthy</h3>
            <h2>{(df['target'] == 0).sum()}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Preview", 
        "🔍 Factor Analysis", 
        "📈 Results", 
        "🎨 Visualizations", 
        "📝 Interpretation"
    ])
    
    with tab1:
        st.subheader("📋 Dataset Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(df.head(20), use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Feature Description")
            st.markdown("""
            - **age**: Age in years
            - **sex**: Male=1, Female=0
            - **cp**: Chest pain type
            - **trestbps**: Resting blood pressure
            - **chol**: Serum cholesterol
            - **fbs**: Fasting blood sugar
            - **restecg**: Resting ECG results
            - **thalach**: Max heart rate achieved
            - **exang**: Exercise induced angina
            - **oldpeak**: ST depression
            - **slope**: Slope of ST segment
            - **ca**: Number of major vessels
            - **thal**: Thalassemia
            - **target**: Heart disease (1=Yes, 0=No)
            """)
        
        # Statistics
        st.subheader("📊 Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab2:
        st.subheader("🔬 Performing Factor Analysis")
        
        # Prepare data
        features = df.drop('target', axis=1)
        
        # Handle missing values
        if features.isnull().sum().sum() > 0:
            st.warning("⚠️ Missing values detected. Filling with median...")
            features = features.fillna(features.median())
        
        # Standardization
        with st.spinner("🔄 Standardizing data..."):
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            features_scaled = pd.DataFrame(features_scaled, columns=features.columns)
        
        st.success("✅ Data standardized successfully!")
        
        # KMO and Bartlett test
        st.subheader("📊 Adequacy Tests")
        
        col1, col2 = st.columns(2)
        
        with col1:
            kmo_all, kmo_model = calculate_kmo(features_scaled)
            st.metric(
                "KMO Test", 
                f"{kmo_model:.3f}",
                help="> 0.6 indicates good factorability"
            )
            if kmo_model > 0.6:
                st.success("✅ KMO > 0.6 - Suitable for factor analysis")
            else:
                st.warning("⚠️ KMO < 0.6 - Data may not be suitable")
        
        with col2:
            chi2, p_value = calculate_bartlett_sphericity(features_scaled)
            st.metric(
                "Bartlett's Test", 
                f"p = {p_value:.6f}",
                help="p < 0.05 indicates correlation matrix is not identity"
            )
            if p_value < 0.05:
                st.success("✅ p < 0.05 - Correlations exist")
            else:
                st.warning("⚠️ p > 0.05 - Data may not be suitable")
        
        # Perform Factor Analysis
        with st.spinner("🧮 Running Factor Analysis..."):
            fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method)
            fa.fit(features_scaled)
        
        st.success(f"✅ Factor Analysis completed with {n_factors} factors!")
        
        # Save results in session state
        st.session_state['fa'] = fa
        st.session_state['features_scaled'] = features_scaled
        st.session_state['features_names'] = features.columns
    
    with tab3:
        if 'fa' in st.session_state:
            fa = st.session_state['fa']
            
            # Eigenvalues
            st.subheader("📊 Eigenvalues")
            ev, v = fa.get_eigenvalues()
            
            # Create dataframe for eigenvalues
            eigen_df = pd.DataFrame({
                'Factor': [f'F{i+1}' for i in range(len(ev))],
                'Eigenvalue': ev,
                'Variance Explained (%)': (ev / ev.sum() * 100)
            })
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.dataframe(eigen_df.head(10), use_container_width=True)
            
            with col2:
                # Components with eigenvalue > 1
                n_components = sum(ev > 1)
                st.metric("Recommended Factors (Eigenvalue > 1)", n_components)
            
            # Factor Loadings
            st.subheader("📊 Factor Loadings Matrix")
            
            loadings = pd.DataFrame(
                fa.loadings_,
                index=st.session_state['features_names'],
                columns=[f'Factor_{i+1}' for i in range(n_factors)]
            )
            
            st.dataframe(loadings.round(3), use_container_width=True)
            
            # Communalities
            st.subheader("📊 Communalities")
            comm = fa.get_communalities()
            comm_df = pd.DataFrame({
                'Feature': st.session_state['features_names'],
                'Communality': comm.round(3)
            }).sort_values('Communality', ascending=False)
            
            st.dataframe(comm_df, use_container_width=True)
            
            # Variance explained
            st.subheader("📊 Variance Explained")
            var_df = pd.DataFrame({
                'Factor': [f'Factor {i+1}' for i in range(n_factors)],
                'SS Loadings': fa.get_factor_variance()[0],
                'Proportion Var': fa.get_factor_variance()[1],
                'Cumulative Var': fa.get_factor_variance()[2]
            })
            
            st.dataframe(var_df, use_container_width=True)
            
            # Download results
            st.subheader("💾 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_loadings = loadings.to_csv()
                st.download_button(
                    "📥 Download Loadings",
                    csv_loadings,
                    "factor_loadings.csv",
                    "text/csv"
                )
            
            with col2:
                csv_communalities = comm_df.to_csv()
                st.download_button(
                    "📥 Download Communalities",
                    csv_communalities,
                    "communalities.csv",
                    "text/csv"
                )
            
            with col3:
                csv_eigen = eigen_df.to_csv()
                st.download_button(
                    "📥 Download Eigenvalues",
                    csv_eigen,
                    "eigenvalues.csv",
                    "text/csv"
                )
    
    with tab4:
        if 'fa' in st.session_state:
            fa = st.session_state['fa']
            ev, v = fa.get_eigenvalues()
            
            st.subheader("🎨 Visualizations")
            
            # Select visualization type
            viz_type = st.selectbox(
                "Select Visualization",
                ["Scree Plot", "Heatmap of Loadings", "Variance Explained", "Communalities Plot", "Correlation Matrix"]
            )
            
            # Scree Plot
            if viz_type == "Scree Plot":
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot with custom colors
                ax.plot(range(1, len(ev)+1), ev, 'o-', 
                       color=color_theme, linewidth=2, markersize=8)
                ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.fill_between(range(1, len(ev)+1), ev, 1, alpha=0.2, color=color_theme)
                
                ax.set_title('Scree Plot - Factor Analysis', fontsize=16, fontweight='bold')
                ax.set_xlabel('Factor Number', fontsize=12)
                ax.set_ylabel('Eigenvalue', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('#f8f9fa')
                
                st.pyplot(fig)
                plt.close()
                
                # Interpretation
                st.info(f"📌 **Interpretation:** Factors with eigenvalue > 1 are retained. Recommended: {sum(ev > 1)} factors.")
            
            # Heatmap
            elif viz_type == "Heatmap of Loadings":
                loadings = pd.DataFrame(
                    fa.loadings_,
                    index=st.session_state['features_names'],
                    columns=[f'Factor_{i+1}' for i in range(n_factors)]
                )
                
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(loadings, annot=True, cmap='RdPu', fmt='.2f', 
                           linewidths=0.5, ax=ax, cbar_kws={'label': 'Loading'})
                ax.set_title('Factor Loadings Heatmap', fontsize=16, fontweight='bold')
                ax.set_xlabel('Factors', fontsize=12)
                ax.set_ylabel('Features', fontsize=12)
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close()
            
            # Variance Explained
            elif viz_type == "Variance Explained":
                var_df = pd.DataFrame({
                    'Factor': [f'F{i+1}' for i in range(n_factors)],
                    'Variance (%)': fa.get_factor_variance()[1] * 100
                })
                
                fig = px.bar(var_df, x='Factor', y='Variance (%)', 
                            title='Variance Explained by Each Factor',
                            color='Variance (%)', color_continuous_scale='RdPu')
                fig.update_layout(showlegend=False, height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Communalities Plot
            elif viz_type == "Communalities Plot":
                comm = fa.get_communalities()
                comm_df = pd.DataFrame({
                    'Feature': st.session_state['features_names'],
                    'Communality': comm
                }).sort_values('Communality', ascending=True)
                
                fig = px.bar(comm_df, x='Communality', y='Feature', 
                            orientation='h', title='Communalities (Variance Explained by Factors)',
                            color='Communality', color_continuous_scale='RdPu')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation Matrix
            elif viz_type == "Correlation Matrix":
                corr_matrix = st.session_state['features_scaled'].corr()
                
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdPu',
                           linewidths=0.5, ax=ax, square=True)
                ax.set_title('Correlation Matrix of Standardized Features', fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close()
    
    with tab5:
        st.subheader("📝 Interpretation Guide")
        
        st.markdown("""
        ### 🎯 **How to Interpret Factor Analysis Results**
        
        #### 1. **KMO and Bartlett's Test**
        - **KMO > 0.6**: Data is suitable for factor analysis
        - **Bartlett's p < 0.05**: Variables are correlated enough for factor analysis
        
        #### 2. **Eigenvalues**
        - **Eigenvalue > 1**: Standard criterion for factor retention
        - **Scree Plot**: Look for the "elbow" point
        
        #### 3. **Factor Loadings**
        - **> |0.4|**: Strong loading (important for the factor)
        - **> |0.3|**: Moderate loading
        - **< |0.3|**: Weak loading
        
        #### 4. **Communalities**
        - **> 0.5**: Good portion of variance explained by factors
        - **< 0.3**: Variable may not be well represented
        
        #### 5. **Variance Explained**
        - **Cumulative > 60%**: Good total variance explained
        """
        )
        
        # Generate interpretation based on results
        if 'fa' in st.session_state:
            st.markdown("### 🔍 **Your Results Interpretation**")
            
            # Get results
            ev, _ = st.session_state['fa'].get_eigenvalues()
            comm = st.session_state['fa'].get_communalities()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✅ **Good Signs**")
                if sum(ev > 1) > 2:
                    st.success(f"✓ {sum(ev > 1)} factors have eigenvalue > 1")
                if np.mean(comm) > 0.5:
                    st.success(f"✓ Average communality: {np.mean(comm):.3f} (> 0.5)")
                st.info("✓ Factor structure is interpretable")
            
            with col2:
                st.markdown("#### ⚠️ **Considerations**")
                if sum(ev > 1) < 3:
                    st.warning("⚠️ Few factors detected")
                if np.mean(comm) < 0.4:
                    st.warning("⚠️ Low average communality")
            
            # Feature recommendations
            st.markdown("### 💡 **Clinical Recommendations**")
            st.markdown("""
            Based on factor analysis results, consider focusing on:
            - **Primary risk factors**: Features with highest loadings on first factor
            - **Secondary factors**: Explore combinations of features within each factor
            - **Screening tools**: Use reduced feature set based on factor loadings
            """)

else:
    # Welcome screen
    st.markdown('<div class="main-title">❤️ Welcome to Factor Analysis App</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h2>📊 Heart Disease Dataset</h2>
            <p style="font-size: 18px;">Upload your CSV file to perform factor analysis</p>
            <p style="color: #666;">🎯 Features:</p>
            <ul style="text-align: left;">
                <li>Automatic data preprocessing</li>
                <li>Interactive visualizations</li>
                <li>Professional factor analysis</li>
                <li>Downloadable results</li>
                <li>Clinical interpretation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Example data
        if st.button("📂 Load Example Data"):
            # Create example from the data provided
            from io import StringIO
            df_example = pd.read_csv('heart.csv')
            st.session_state['example_df'] = df_example
            st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Developed with ❤️ for Medical Data Analysis | Factor Analysis for Heart Disease Prediction</p>
</div>
""", unsafe_allow_html=True)