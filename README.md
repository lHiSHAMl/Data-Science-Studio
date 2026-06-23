# Data Science Studio (DSS)

## Overview

**Data Science Studio** is an integrated, no-code web-based data science platform designed to simplify and streamline the entire data science lifecycle. Built with Python and Streamlit, DSS enables users to perform data preprocessing, visualization, statistical analysis, machine learning, and reporting all within a single unified interface—without requiring extensive programming knowledge.

The platform addresses critical pain points in traditional data science workflows:
- **Fragmented toolchains** that require switching between multiple platforms
- **Repetitive coding** for common data science tasks
- **High barriers to entry** for non-technical users and beginners
- **Unstructured workflow management** across disconnected tools
- **Slow iteration cycles** when experimenting with different approaches

### Key Features

✨ **Unified Workflow** - Complete data science pipeline from raw data to final report in one platform

🔧 **No-Code Interface** - Perform complex operations without writing code

📊 **Interactive Visualizations** - 15+ chart types with advanced configuration options

🤖 **Intelligent AI Advisory** - Automated dataset profiling, risk detection, and model recommendations

📈 **Machine Learning Suite** - Supervised and unsupervised models with automated evaluation

💾 **Portable Projects** - JSON-based save files for lightweight, cross-device portability

📋 **Built-in Reporting** - Collect and export professional reports in PDF/HTML

---

## System Architecture

### Core Components

1. **Data Transformations Module**
   - Dynamic data preprocessing and feature engineering
   - 6 transformation categories: Cleaning, Basic, Standardization, Dimensionality Reduction, Encoding, Feature Selection
   - Non-destructive, reusable transformation pipelines

2. **Visualization & EDA Module**
   - Distribution, relationship, and statistical plots
   - Advanced charts: heatmaps, 3D scatter, parallel coordinates, treemaps
   - Two-level transformation pipeline (global + per-chart)
   - Multiple export formats: PNG, SVG, PDF, interactive HTML

3. **Statistical Analysis Module**
   - Exploratory data analysis tools
   - Form-based configuration for hypothesis tests
   - Interpretive summaries for accessible statistical insights

4. **Machine Learning Module**
   - **Classification Models**: XGBoost, Gradient Boosting, Random Forest, Logistic Regression, SVM, Naive Bayes
   - **Regression Models**: Linear, Ridge, Lasso, KNN, Decision Tree, Random Forest, XGBoost
   - **Unsupervised Learning**: K-Means, Hierarchical Clustering, DBSCAN
   - Model descriptions, reference code, configurable hyperparameters
   - Automated evaluation reports with metrics and visualizations
   - Model export to `.pkl` for external deployment

5. **AI Advisory System**
   - **Intelligent Data Profiling**: Automatic ML task detection, statistical risk analysis, data quality assessment
   - **Dual-AI Architecture**: 
     - Planner (deepseek-r1) for technical reasoning
     - Writer (llama3.2) for professional communication
   - **Risk Detection**: Data leakage, class imbalance, multicollinearity, quality issues
   - **Automated Recommendations**: Data cleaning, feature engineering, model selection
   - **Deployment Guidance**: Pre-deployment validation checklist, monitoring strategy

6. **Reporting Module**
   - Collect outputs from any module as report items
   - Reusable artifacts: transformed datasets, ML evaluations, visualizations
   - Drag-and-drop reordering and selective inclusion
   - Export to PDF or interactive HTML

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **Backend** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly |
| **Machine Learning** | scikit-learn, XGBoost |
| **Dimensionality Reduction** | scikit-learn, UMAP, t-SNE |
| **PDF Generation** | ReportLab, FPDF2 |
| **Local LLMs** | Ollama (deepseek-r1, llama3.2) |
| **Advanced AI** | Google Gemini API (optional) |

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- Modern web browser

### Step 1: Clone the Repository
```bash
git clone https://github.com/lHiSHAMl/Data-Science-Studio.git
cd Data-Science-Studio
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: (Optional) Setup Ollama for AI Advisory
For full AI Advisory System functionality:
```bash
# Download and install Ollama from https://ollama.ai
ollama pull deepseek-r1:1.5b
ollama pull llama3.2:1b
```

### Step 5: Run the Application
```bash
streamlit run app.py
```

The application will start at `http://localhost:8501`

---

## Usage Guide

### Basic Workflow

1. **Upload Dataset**
   - Select CSV, XLSX, or JSON file from your device
   - Or choose a sample dataset from the built-in repository

2. **Apply Transformations**
   - Navigate to Data Transformations
   - Create preprocessing steps (cleaning, encoding, scaling, etc.)
   - Preview transformed data in real-time

3. **Explore Data**
   - Use Visualizations module to create charts
   - Apply transformations specific to each visualization
   - Add insights to your project notes

4. **Statistical Analysis**
   - Run hypothesis tests
   - Generate correlation matrices
   - Assess data distributions

5. **Machine Learning**
   - Select target variable and features
   - Choose a model from available options
   - Configure hyperparameters
   - Train and evaluate performance

6. **Generate Reports**
   - Collect key outputs as report items
   - Arrange in logical order
   - Export as PDF or HTML

7. **Save & Share**
   - Save project to lightweight JSON file
   - Resume work later on any device
   - Share project file with team members

### Example: Classification Pipeline

```
1. Upload dataset with customer churn data
2. Transform: Handle missing values → Encode categorical vars → Scale features
3. Visualize: Class distribution → Feature correlations → ROC curves
4. ML: Select XGBoost Classifier → Configure hyperparameters → Train model
5. Evaluate: Review confusion matrix, precision, recall, F1-score
6. Report: Add visualizations + evaluation metrics → Export PDF
7. Deploy: Export trained model as .pkl file
```


---
**Last Updated:** June 2026
**Current Version:** 1.0.0
**Status:** Active Development
