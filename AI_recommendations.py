import streamlit as st
import pandas as pd
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from typing import Dict, Any, Optional
import re
import warnings
from openai import OpenAI  # Updated import for v1+ client
warnings.filterwarnings("ignore")

# Constants
SYSTEM_PROMPT = "You are an expert AI data science advisor. Provide structured, visual-friendly markdown output."
MODEL_NAME = "gpt-4o-mini"

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main-header {font-size: 3rem; color: #2e86de; text-align: center; margin-bottom: 2rem;}
    .section-header {font-size: 2.2rem; color: #f39c12; border-bottom: 2px solid #f39c12; padding-bottom: 0.5rem;}
    .recommendation-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0;}
    .metric-box {background: #e8f4fd; padding: 1rem; border-radius: 8px; border-left: 5px solid #3498db;}
    .model-comparison {background: #f8f9fa; padding: 1rem; border-radius: 8px;}
    
    /* Horizontal tabs styling */
    .recommendation-card{
            background: black;
            
            }
    .horizontal-tabs {
        display: flex;
        gap: 10px;
        margin-bottom: 2rem;
        justify-content: center;
    }
    .tab-button {
        padding: 12px 24px;
        border: 2px solid #2e86de;
        border-radius: 8px;
        background: white;
        color: #2e86de;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .tab-button:hover {
        background: #2e86de;
        color: white;
    }
    .tab-button.active {
        background: #2e86de;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize OpenAI Client (Secure via Secrets) ---
@st.cache_resource
def init_openai():
    """Initialize OpenAI client with API key from secrets."""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=api_key)
        # Test connection with a simple call
        client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": "test"}], max_tokens=1)
        return client
    except Exception as e:
        st.error(f"❌ OpenAI setup failed: {str(e)}. Ensure OPENAI_API_KEY is set in secrets.toml (local) or Streamlit Cloud settings.")
        if "No secrets found" in str(e):
            st.info("💡 **Local Fix**: Create `.streamlit/secrets.toml` with `OPENAI_API_KEY = 'sk-your-key'`. See docs.")
        return None

# --- Data Handling with Session State ---
def load_dataset(uploaded_file: Any) -> pd.DataFrame:
    """Load dataset from uploaded file and store in session state."""
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file type.")
        
        # Store in session state for cross-tab persistence
        st.session_state.current_dataset = df
        st.session_state.dataset_name = uploaded_file.name
        st.session_state.dataset_loaded = True
        
        # Clear recommendations if dataset changes
        if 'current_dataset_hash' not in st.session_state or st.session_state.current_dataset_hash != hash(str(df.shape) + uploaded_file.name):
            st.session_state.recommendations = None
            st.session_state.current_dataset_hash = hash(str(df.shape) + uploaded_file.name)
            
        return df
    except Exception as e:
        st.error(f"❌ Failed to load: {str(e)}")
        return pd.DataFrame()

def get_current_dataset() -> Optional[pd.DataFrame]:
    """Get the current dataset from session state."""
    return st.session_state.get('current_dataset', None)

def generate_summary(df: pd.DataFrame) -> str:
    """Generate concise summary for LLM."""
    if df.empty:
        return "No data."
    
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    desc = df.describe(include='all').round(2).to_string()
    missing = df.isnull().sum().to_string()
    
    return f"""
Shape: {df.shape}
Columns: {list(df.columns)}
Info: {info_str}
Missing: {missing}
Desc: {desc}
    """

# --- LLM Query (Updated for OpenAI v1+ Client) ---
@st.cache_data
def query_llm(prompt: str) -> Optional[str]:
    """Query OpenAI API using client."""
    client = init_openai()
    if not client:
        return None
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"❌ LLM API error: {str(e)}. Check API key/credits/quota.")
        return None

def generate_enhanced_recommendations(summary: str, df: pd.DataFrame, task_type: str = "Classification") -> Optional[str]:
    """Generate recommendations with visual cues in prompt."""
    prompt = f"""
You are a senior Data Science Consultant. For this {task_type} dataset:

Summary:
{summary}

Provide a **structured markdown report** optimized for visualization and decision-making. Use:

- Headers for sections.
- Bullet points/tables for clarity.
- Emojis: 📊 for data, 🔧 for prep, 🤖 for models, 📈 for eval, 🚀 for deployment.
- Suggest visuals: e.g., "Visualize with correlation heatmap 📊" or "Compare models in a bar chart of accuracy vs. speed 🤖".
- For models: Table with columns: Model, Pros, Cons, Best For, Expected Performance (e.g., F1: 0.85-0.95).
- For metrics: Explain with examples (e.g., "F1-score balances precision/recall for imbalanced data 📈").
- Keep concise (500-1000 words), actionable, evidence-based.

Sections:
1️⃣ **Data Understanding** 📊
2️⃣ **Data Transformations** 🔧
3️⃣ **Model Recommendations** 🤖 (include comparison table)
4️⃣ **Evaluation Strategy** 📈 (include metric visuals ideas)
5️⃣ **Next Steps** 🚀
    """
    return query_llm(prompt)

# --- Visual Enhancers ---
def create_model_comparison_visual(models_data: Dict[str, Any]) -> go.Figure:
    """Create a radar chart for model comparison."""
    fig = go.Figure()
    
    for model, metrics in models_data.items():
        fig.add_trace(go.Scatterpolar(
            r=list(metrics.values()),
            theta=list(metrics.keys()),
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Comparison Radar Chart 🤖<br>(Higher values = Better Performance)",
        height=500
    )
    return fig

def create_preprocessing_flowchart() -> go.Figure:
    """Sankey diagram for typical preprocessing pipeline."""
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Raw Data", "Handle Missing", "Encode Cats", "Scale Numerics", "Feature Eng", "Clean Dataset"],
            color=["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
        ),
        link=dict(
            source=[0, 0, 1, 2, 3, 4],
            target=[1, 2, 3, 4, 5, 5],
            value=[1, 1, 1, 1, 1, 1]
        )
    ))
    fig.update_layout(title="Recommended Preprocessing Pipeline 🔧", height=400, font_size=12)
    return fig

def create_metrics_visual(task_type: str) -> go.Figure:
    """Bar chart explaining key metrics."""
    if task_type == "Classification":
        metrics = {"Accuracy": 0.85, "Precision": 0.88, "Recall": 0.82, "F1-Score": 0.85}
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    else:  # Regression
        metrics = {"MAE": 0.15, "MSE": 0.25, "RMSE": 0.5, "R²": 0.92}
        colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    
    fig = px.bar(x=list(metrics.keys()), y=list(metrics.values()), color=list(metrics.keys()),
                 color_discrete_sequence=colors, title=f"Key {task_type} Metrics 📈<br>(Example Values for Decision-Making)")
    fig.update_layout(height=400, xaxis_title="Metrics", yaxis_title="Score/Value")
    return fig

def parse_and_enhance_recommendations(recommendations: str, task_type: str) -> Dict[str, Any]:
    """Parse LLM output and prepare for enhanced display."""
    sections = re.split(r'(?=^\d+️⃣ .+)', recommendations, flags=re.MULTILINE)
    enhanced = {"full_text": recommendations, "sections": sections}
    
    enhanced["visuals"] = {
        "model_radar": create_model_comparison_visual({
            "Random Forest": {"Accuracy": 0.92, "Speed": 0.75, "Interpretability": 0.85},
            "XGBoost": {"Accuracy": 0.95, "Speed": 0.60, "Interpretability": 0.40},
            "Neural Net": {"Accuracy": 0.93, "Speed": 0.30, "Interpretability": 0.20}
        }),
        "prep_flow": create_preprocessing_flowchart(),
        "metrics_bar": create_metrics_visual(task_type)
    }
    
    return enhanced

# --- Main Run Function ---
def run():
    """Main function to run the AI Recommendations tab."""
    
    # Initialize Session State
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'current_task_type' not in st.session_state:
        st.session_state.current_task_type = "Classification"

    # Main UI
    st.markdown('<h1 class="main-header">🚀 DSS Recommendations Hub</h1>', unsafe_allow_html=True)
    st.markdown("Focus on **AI-powered insights** with embedded visuals to guide your data science decisions. Upload → Generate → Visualize & Choose.Powered by OpenAI to ease your decision making.")

    # Show current dataset info if available
    if st.session_state.dataset_loaded and st.session_state.current_dataset is not None:
        st.success(f"📊 **Loaded Dataset:** {st.session_state.dataset_name}")
        st.write(f"**Dataset Shape:** {st.session_state.current_dataset.shape}")

    # File upload
    uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xlsx", "xls"])
    
    # Use existing dataset or load new one
    current_df = None
    if uploaded_file is not None:
        # New file uploaded
        current_df = load_dataset(uploaded_file)
    elif st.session_state.dataset_loaded and st.session_state.current_dataset is not None:
        # Use existing dataset from session state
        current_df = st.session_state.current_dataset
        st.info(f"📁 Using previously loaded dataset: **{st.session_state.dataset_name}**")
    
    if current_df is None or current_df.empty:
        st.info("👆 Upload a dataset to begin.")
        return

    # Display current dataset
    st.caption(f"✅ Loaded {current_df.shape[0]} rows and {current_df.shape[1]} columns.")
    
    with st.expander("📋 View Dataset Preview", expanded=False):
        st.dataframe(current_df.head(), use_container_width=True)

    # Dataset summary
    with st.expander("📊 Quick Summary", expanded=False):
        buffer = io.StringIO()
        current_df.info(buf=buffer)
        st.text(buffer.getvalue())

    # Clear recommendations button
    if st.session_state.recommendations is not None:
        if st.button("🗑️ Clear Recommendations", type="secondary"):
            st.session_state.recommendations = None
            st.rerun()

    # Generate recommendations
    if st.button("🔮 Generate Enhanced Recommendations", type="primary", use_container_width=True):
        with st.spinner("🤖 AI analyzing your dataset... (10-30 seconds)"):
            summary = generate_summary(current_df)
            rec_text = generate_enhanced_recommendations(summary, current_df, st.session_state.current_task_type)
            
            if rec_text:
                st.session_state.recommendations = parse_and_enhance_recommendations(rec_text, st.session_state.current_task_type)
                st.success("✅ Recommendations generated successfully!")
            else:
                st.error("❌ Failed to generate recommendations. Check API setup.")
                st.session_state.recommendations = None

    # Display recommendations
    if st.session_state.recommendations is not None and isinstance(st.session_state.recommendations, dict):
        enhanced = st.session_state.recommendations
        st.markdown("---")
        st.markdown('<h2 class="section-header">🧭 Enhanced Recommendations</h2>', unsafe_allow_html=True)

        # Render full text with markdown
        st.markdown(enhanced["full_text"])

        # Embedded Visuals
        st.markdown("---")
        st.markdown('<h3 class="section-header">📊 Visual Aids for Better Decisions</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(enhanced["visuals"]["prep_flow"], use_container_width=True)
            st.markdown("**🔧 Preprocessing Flow**: Follow this pipeline to transform your data efficiently.")

        with col2:
            st.plotly_chart(enhanced["visuals"]["metrics_bar"], use_container_width=True)
            st.markdown("**📈 Key Metrics**: Use these to evaluate and compare models. Higher is better for most.")

        st.plotly_chart(enhanced["visuals"]["model_radar"], use_container_width=True)
        st.markdown("**🤖 Model Comparison**: Radar chart shows trade-offs (Accuracy, Speed, Interpretability). Choose based on priorities.")

        # Download option
        st.download_button(
            "💾 Download Full Report (Markdown)",
            data=enhanced["full_text"],
            file_name="llama_recommendations.md",
            mime="text/markdown",
            use_container_width=True
        )

        st.info("✨ Visuals help prioritize: e.g., High interpretability for explainable AI needs.")
    elif st.session_state.recommendations is None and current_df is not None:
        st.info("👆 Click 'Generate Enhanced Recommendations' to get AI-powered insights with visuals.")

# For direct execution testing
if __name__ == "__main__":
    run()