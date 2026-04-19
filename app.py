import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import joblib
import os
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Batang Quiapo | Iris ML",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Fresh Light Botanical Theme ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg:       #f5f0eb;
    --white:    #ffffff;
    --card:     #ffffff;
    --green1:   #2d6a4f;
    --green2:   #52b788;
    --green3:   #d8f3dc;
    --pink1:    #e07a5f;
    --pink2:    #f2cc8f;
    --text:     #1a1a2e;
    --muted:    #6b7280;
    --border:   #e8e0d5;
    --shadow:   0 4px 24px rgba(45,106,79,0.10);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--green1) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #fff !important; }
[data-testid="stSidebar"] .stRadio label { color: #d8f3dc !important; font-size: 0.9rem !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
    color: #fff !important; font-weight: 700 !important;
}
[data-testid="stSidebar"] .stSlider label { color: #d8f3dc !important; }

h1,h2,h3,h4,h5 {
    font-family: 'Fraunces', serif !important;
    color: var(--green1) !important;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #2d6a4f 0%, #40916c 60%, #52b788 100%);
    border-radius: 24px;
    padding: 2.8rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(45,106,79,0.25);
}
.hero::after {
    content: '🌿';
    position: absolute;
    font-size: 160px;
    right: 2rem; bottom: -1rem;
    opacity: 0.15;
    line-height: 1;
}
.hero-label {
    background: rgba(255,255,255,0.2);
    color: #d8f3dc;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    display: inline-block;
    margin-bottom: 0.8rem;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: 3rem;
    font-weight: 900;
    color: #ffffff;
    line-height: 1.1;
    margin: 0 0 0.5rem;
}
.hero-desc { color: #b7e4c7; font-size: 1rem; margin: 0; }

/* Cards */
.card {
    background: var(--card);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
}
.card-title {
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    color: var(--green1);
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}

/* Step badge */
.step-pill {
    background: var(--green1);
    color: #fff;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.25rem 0.8rem;
    border-radius: 50px;
    display: inline-block;
    margin-bottom: 0.6rem;
}

/* Member cards */
.member-card {
    background: linear-gradient(135deg, #f0faf4, #fff);
    border: 1.5px solid #b7e4c7;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.85rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 2px 12px rgba(45,106,79,0.07);
    transition: transform 0.2s;
}
.member-avatar {
    width: 42px; height: 42px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--green1), var(--green2));
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 0.95rem; color: #fff;
    flex-shrink: 0;
    box-shadow: 0 3px 10px rgba(45,106,79,0.3);
}
.member-name { font-size: 0.9rem; font-weight: 600; color: var(--text); }
.member-role { font-size: 0.72rem; color: var(--green2); font-weight: 500; }
.member-num  { font-size: 0.68rem; color: #b0b8c1; }

/* Metric chip */
.metric-chip {
    background: linear-gradient(135deg, #f0faf4, #e8f5f0);
    border: 1.5px solid #b7e4c7;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(45,106,79,0.07);
}
.metric-val {
    font-family: 'Fraunces', serif;
    font-size: 2.2rem;
    font-weight: 900;
    color: var(--green1);
}
.metric-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.1rem; }

/* Pipeline step */
.pipeline-step {
    background: #fff;
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 1rem 0.8rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(45,106,79,0.05);
}
.pipeline-icon { font-size: 2rem; margin-bottom: 0.3rem; }
.pipeline-num  { font-size: 0.65rem; color: var(--green2); font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; }
.pipeline-name { font-family: 'Fraunces', serif; font-size: 0.9rem; color: var(--green1); font-weight: 700; }
.pipeline-desc { font-size: 0.7rem; color: var(--muted); margin-top: 0.3rem; }

/* Prediction box */
.pred-box {
    background: linear-gradient(135deg, #f0faf4, #e8f5f0);
    border: 2px solid var(--green2);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(82,183,136,0.15);
}
.pred-emoji   { font-size: 4rem; }
.pred-species {
    font-family: 'Fraunces', serif;
    font-size: 2.2rem;
    color: var(--green1);
    font-weight: 900;
    margin: 0.3rem 0;
}
.pred-conf { color: var(--muted); font-size: 0.9rem; }

/* Info / success boxes */
.info-box {
    background: #e8f4fd;
    border-left: 4px solid #3b82f6;
    border-radius: 0 12px 12px 0;
    padding: 0.85rem 1rem;
    font-size: 0.88rem;
    color: #1e3a5f;
    margin: 0.6rem 0;
}
.success-box {
    background: #f0faf4;
    border-left: 4px solid var(--green2);
    border-radius: 0 12px 12px 0;
    padding: 0.85rem 1rem;
    font-size: 0.88rem;
    color: var(--green1);
    margin: 0.6rem 0;
}
.warning-box {
    background: #fff8ec;
    border-left: 4px solid var(--pink1);
    border-radius: 0 12px 12px 0;
    padding: 0.85rem 1rem;
    font-size: 0.88rem;
    color: #7c4a1e;
    margin: 0.6rem 0;
}

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, var(--green1), var(--green2)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    padding: 0.65rem 2.2rem !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 15px rgba(45,106,79,0.3) !important;
    transition: all 0.2s !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(45,106,79,0.4) !important;
}

[data-testid="stMetricValue"] { color: var(--green1) !important; font-family: 'Fraunces', serif !important; }

div[data-testid="stSelectbox"] > div > div {
    background: #fff !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
}

/* Code blocks */
.stCodeBlock { border-radius: 12px !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 12px !important; }

/* Divider */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "iris_knn_model.joblib"
SCALER_PATH = "iris_scaler.joblib"

MEMBERS = [
    {"name": "John Loise Ibeng",       "role": "Member 1"},
    {"name": "Jaby Maverick Lasquite", "role": "Member 2"},
    {"name": "Rodel Lobendino",        "role": "Member 3"},
    {"name": "Reymark Morales",        "role": "Member 4"},
    {"name": "Jomel Onido",            "role": "Member 5"},
]

SPECIES_EMOJI = {"setosa": "🌷", "versicolor": "🌺", "virginica": "🌸"}
SPECIES_COLOR = ["#2d6a4f", "#52b788", "#e07a5f"]

def get_initials(name):
    p = name.split()
    return (p[0][0] + p[-1][0]).upper()

# ── Data / Model helpers ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return iris, df

def train_and_save(k=5):
    iris, _ = load_data()
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_tr_sc, y_tr)
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler, X_tr, X_te, X_tr_sc, X_te_sc, y_tr, y_te

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    return None, None

iris_data, df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 0.5rem;text-align:center;'>
        <div style='font-size:2.5rem;'>🌸</div>
        <div style='font-family:Fraunces,serif;font-size:1.3rem;
                    color:#fff;font-weight:900;margin:0.3rem 0 0.1rem;'>
            Batang Quiapo
        </div>
        <div style='color:#b7e4c7;font-size:0.7rem;
                    letter-spacing:0.2em;text-transform:uppercase;'>
            Iris ML Classifier
        </div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.15);margin:0.8rem 0;'>
    """, unsafe_allow_html=True)

    nav = st.radio("Go to", [
        "🏠  Home",
        "📊  Step 1: Load Data",
        "🔍  Step 2: Explore",
        "⚙️  Step 3: Train",
        "📈  Step 4: Evaluate",
        "🎯  Step 5: Predict",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='color:rgba(255,255,255,0.4);font-size:0.7rem;text-align:center;'>KNN · scikit-learn · joblib<br>Streamlit · Python</div>", unsafe_allow_html=True)

k_val = 5  # default K value for KNN

# ═══════════════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════════════
if nav == "🏠  Home":
    st.markdown("""
    <div class='hero'>
        <div class='hero-label'>🌿 Machine Learning Project</div>
        <div class='hero-title'>Iris Flower<br>Classifier</div>
        <div class='hero-desc'>Step-by-step ML pipeline · KNN · scikit-learn · joblib · Streamlit</div>
    </div>
    """, unsafe_allow_html=True)

    # Group name
    st.markdown("""
    <div class='card'>
        <div style='display:flex;align-items:center;gap:1rem;flex-wrap:wrap;'>
            <div style='font-size:2.5rem;'>👥</div>
            <div>
                <div style='font-size:0.72rem;color:#52b788;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.15em;'>Group Name</div>
                <div style='font-family:Fraunces,serif;font-size:2rem;
                            color:#2d6a4f;font-weight:900;'>Batang Quiapo</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Members
    st.markdown("<div class='card'><div class='card-title'>👤 Group Members</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for i, m in enumerate(MEMBERS):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"""
            <div class='member-card'>
                <div class='member-avatar'>{get_initials(m['name'])}</div>
                <div>
                    <div class='member-num'>Member {i+1}</div>
                    <div class='member-name'>{m['name']}</div>
                    <div class='member-role'>🌿 {m['role']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Quick stats
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip([c1,c2,c3,c4],
                                ["150","4","3","KNN"],
                                ["Samples","Features","Classes","Algorithm"]):
        with col:
            st.markdown(f"<div class='metric-chip'><div class='metric-val'>{val}</div><div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline overview
    st.markdown("<div class='card'><div class='card-title'>🗺️ ML Pipeline</div>", unsafe_allow_html=True)
    steps = [
        ("📊","1","Load Data",   "Load Iris from sklearn"),
        ("🔍","2","Explore",     "Visualize patterns"),
        ("⚙️","3","Train",       "KNN + joblib save"),
        ("📈","4","Evaluate",    "Accuracy & metrics"),
        ("🎯","5","Predict",     "Classify new flowers"),
    ]
    cols = st.columns(5)
    for col, (icon, num, name, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class='pipeline-step'>
                <div class='pipeline-icon'>{icon}</div>
                <div class='pipeline-num'>Step {num}</div>
                <div class='pipeline-name'>{name}</div>
                <div class='pipeline-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "📊  Step 1: Load Data":
    st.markdown("""
    <div class='hero'>
        <div class='step-pill'>STEP 1</div>
        <div class='hero-title' style='font-size:2.2rem;'>Load Dataset</div>
        <div class='hero-desc'>Loading the Iris dataset from scikit-learn</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    📌 <b>Iris Dataset</b> — 150 samples ng iris flowers na may 4 features:
    <b>sepal length</b>, <b>sepal width</b>, <b>petal length</b>, <b>petal width</b>.
    I-classify sa 3 species: setosa, versicolor, at virginica.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🐍 Code")
    st.code("""from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
""", language="python")

    st.markdown("### 📋 Dataset Preview (150 rows)")
    st.dataframe(df, use_container_width=True, height=400)

    c1, c2, c3 = st.columns(3)
    for col, val, lbl in zip([c1,c2,c3], [150, 4, 3], ["Rows","Features","Classes"]):
        with col:
            st.markdown(f"<div class='metric-chip'><div class='metric-val'>{val}</div><div class='metric-label'>{lbl}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor='#ffffff')
    dist = df['species'].value_counts()
    colors = SPECIES_COLOR
    bars = ax.bar(dist.index, dist.values, color=colors, width=0.5, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, dist.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', color='#1a1a2e', fontsize=11, fontweight='bold')
    ax.set_facecolor('#f5f0eb')
    fig.patch.set_facecolor('#ffffff')
    ax.tick_params(colors='#6b7280')
    ax.spines[:].set_color('#e8e0d5')
    ax.set_ylabel('Count', color='#6b7280', fontsize=9)
    ax.set_title('Class Distribution', color='#2d6a4f', fontfamily='serif', fontsize=11)
    st.pyplot(fig, use_container_width=False)

    st.markdown("<div class='success-box'>✅ Dataset loaded! 150 samples, 4 features, 3 classes — ready!</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: EXPLORE
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "🔍  Step 2: Explore":
    st.markdown("""
    <div class='hero'>
        <div class='step-pill'>STEP 2</div>
        <div class='hero-title' style='font-size:2.2rem;'>Exploratory Analysis</div>
        <div class='hero-desc'>Visualizing patterns, distributions, and correlations</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📦 Box Plots", "🔥 Correlation", "⚫ Scatter Plot"])
    palette = {"setosa": "#2d6a4f", "versicolor": "#52b788", "virginica": "#e07a5f"}

    with tab1:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), facecolor='#ffffff')
        fig.patch.set_facecolor('#ffffff')
        for ax, col in zip(axes.flat, df.columns[:-1]):
            data_per = [df[df['species']==sp][col].values for sp in palette]
            bp = ax.boxplot(data_per, patch_artist=True,
                            medianprops=dict(color='white', linewidth=2.5),
                            whiskerprops=dict(color='#b7e4c7', linewidth=1.5),
                            capprops=dict(color='#b7e4c7', linewidth=1.5),
                            flierprops=dict(marker='o', markersize=4, alpha=0.5))
            for patch, color in zip(bp['boxes'], SPECIES_COLOR):
                patch.set_facecolor(color)
                patch.set_alpha(0.85)
            ax.set_facecolor('#f5f0eb')
            ax.set_title(col.replace(' (cm)', ''), color='#2d6a4f', fontsize=9, fontweight='bold')
            ax.set_xticks([1,2,3])
            ax.set_xticklabels(['setosa','versicolor','virginica'], fontsize=7.5, color='#6b7280')
            ax.tick_params(colors='#6b7280')
            ax.spines[:].set_color('#e8e0d5')
        legend_patches = [mpatches.Patch(color=c, label=s) for s, c in zip(palette, SPECIES_COLOR)]
        fig.legend(handles=legend_patches, loc='lower center', ncol=3,
                   fontsize=8, frameon=False)
        plt.tight_layout(pad=2, rect=[0,0.05,1,1])
        st.pyplot(fig, use_container_width=True)

    with tab2:
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='#ffffff')
        corr = df.drop('species', axis=1).corr()
        sns.heatmap(corr, annot=True, fmt=".2f",
                    cmap=sns.light_palette("#2d6a4f", as_cmap=True),
                    ax=ax, linewidths=1, linecolor='#f5f0eb',
                    annot_kws={"size": 11, "weight": "bold"}, square=True)
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#6b7280', labelsize=8)
        ax.set_title('Pearson Correlation Matrix', color='#2d6a4f',
                     fontfamily='serif', fontsize=11, pad=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with tab3:
        feat_cols = list(df.columns[:-1])
        c1, c2 = st.columns(2)
        with c1: x_feat = st.selectbox("X-axis", feat_cols, index=0)
        with c2: y_feat = st.selectbox("Y-axis", feat_cols, index=2)
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='#ffffff')
        for (sp, color) in zip(palette, SPECIES_COLOR):
            sub = df[df['species'] == sp]
            ax.scatter(sub[x_feat], sub[y_feat], color=color,
                       label=sp.capitalize(), alpha=0.85, s=70,
                       edgecolors='white', linewidths=0.8)
        ax.set_facecolor('#f5f0eb')
        ax.set_xlabel(x_feat, color='#6b7280', fontsize=9)
        ax.set_ylabel(y_feat, color='#6b7280', fontsize=9)
        ax.tick_params(colors='#6b7280')
        ax.spines[:].set_color('#e8e0d5')
        ax.legend(facecolor='#fff', edgecolor='#e8e0d5', fontsize=9)
        ax.set_title('Scatter Plot', color='#2d6a4f', fontfamily='serif', fontsize=11)
        st.pyplot(fig, use_container_width=True)

    st.markdown("#### 📋 Descriptive Statistics")
    st.dataframe(df.describe().round(3), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "⚙️  Step 3: Train":
    st.markdown(f"""
    <div class='hero'>
        <div class='step-pill'>STEP 3</div>
        <div class='hero-title' style='font-size:2.2rem;'>Train the Model</div>
        <div class='hero-desc'>Split → Scale → Train KNN (k={k_val}) → Save with joblib</div>
    </div>
    """, unsafe_allow_html=True)

    st.code(f"""from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

X, y = iris.data, iris.target

# 1. Split (80% train / 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# 3. Train KNN (k={k_val})
model = KNeighborsClassifier(n_neighbors={k_val})
model.fit(X_train_sc, y_train)

# 4. Save model + scaler
joblib.dump(model,  'iris_knn_model.joblib')
joblib.dump(scaler, 'iris_scaler.joblib')
""", language="python")

    c1, c2, c3 = st.columns(3)
    for col, v, l in zip([c1,c2,c3], ["150","120","30"], ["Total","Train (80%)","Test (20%)"]):
        with col:
            st.markdown(f"<div class='metric-chip'><div class='metric-val'>{v}</div><div class='metric-label'>{l}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    sub_steps = [
        ("📂 Splitting dataset (80/20)...",       0.25),
        ("📏 Fitting StandardScaler...",           0.50),
        (f"🤖 Training KNN (k={k_val})...",        0.80),
        ("💾 Saving model with joblib...",         1.00),
    ]

    if st.button("🚀 Train & Save Model"):
        prog   = st.progress(0)
        status = st.empty()
        for msg, pct in sub_steps:
            status.markdown(f"<div class='info-box'>⏳ {msg}</div>", unsafe_allow_html=True)
            time.sleep(0.5)
            prog.progress(pct)

        model, scaler, *_, y_te = train_and_save(k_val)
        X_te_sc = scaler.transform(load_iris().data[train_test_split(
            np.arange(150), test_size=0.2, random_state=42, stratify=load_iris().target)[1]])
        y_pred = model.predict(X_te_sc)
        acc = accuracy_score(load_iris().target[train_test_split(
            np.arange(150), test_size=0.2, random_state=42,
            stratify=load_iris().target)[1]], y_pred)

        status.empty(); prog.empty()
        st.markdown(f"""
        <div class='success-box'>
        ✅ <b>Model trained and saved!</b><br>
        &nbsp;&nbsp;📄 iris_knn_model.joblib<br>
        &nbsp;&nbsp;📄 iris_scaler.joblib<br>
        &nbsp;&nbsp;🎯 Test Accuracy: <b>{acc*100:.1f}%</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        if os.path.exists(MODEL_PATH):
            st.markdown("<div class='success-box'>✅ Saved model found! Click above to retrain with new K.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='warning-box'>⚠️ Walang saved model pa. Click ang button para mag-train!</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "📈  Step 4: Evaluate":
    st.markdown("""
    <div class='hero'>
        <div class='step-pill'>STEP 4</div>
        <div class='hero-title' style='font-size:2.2rem;'>Model Evaluation</div>
        <div class='hero-desc'>Confusion matrix, accuracy score, at classification report</div>
    </div>
    """, unsafe_allow_html=True)

    model, scaler = load_model()
    if model is None:
        st.markdown("<div class='warning-box'>⚠️ Wala pang trained model! Pumunta sa <b>Step 3: Train</b> muna.</div>", unsafe_allow_html=True)
        st.stop()

    iris_d = load_iris()
    X, y = iris_d.data, iris_d.target
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_te_sc = scaler.transform(X_te)
    y_pred  = model.predict(X_te_sc)
    acc     = accuracy_score(y_te, y_pred)
    report  = classification_report(y_te, y_pred,
                                    target_names=iris_d.target_names,
                                    output_dict=True)

    # Big accuracy
    st.markdown(f"""
    <div style='text-align:center;padding:1.5rem 0;'>
        <div style='font-size:0.8rem;color:#52b788;font-weight:700;
                    text-transform:uppercase;letter-spacing:0.2em;margin-bottom:0.4rem;'>
            Overall Accuracy
        </div>
        <div style='font-family:Fraunces,serif;font-size:5rem;
                    color:#2d6a4f;font-weight:900;line-height:1;'>
            {acc*100:.1f}%
        </div>
        <div style='color:#6b7280;font-size:0.88rem;margin-top:0.3rem;'>
            Tested on {len(y_te)} samples
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, val, lbl in zip([c1,c2,c3],
        [report['macro avg']['precision'],
         report['macro avg']['recall'],
         report['macro avg']['f1-score']],
        ["Avg Precision","Avg Recall","Avg F1-Score"]):
        with col:
            st.markdown(f"<div class='metric-chip'><div class='metric-val'>{val:.2f}</div><div class='metric-label'>{lbl}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🔲 Confusion Matrix")
        cm = confusion_matrix(y_te, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#ffffff')
        sns.heatmap(cm, annot=True, fmt='d',
                    cmap=sns.light_palette("#2d6a4f", as_cmap=True),
                    xticklabels=iris_d.target_names,
                    yticklabels=iris_d.target_names,
                    ax=ax, linewidths=1.5, linecolor='#f5f0eb',
                    annot_kws={"size": 16, "weight": "bold", "color": "#1a1a2e"})
        ax.set_xlabel("Predicted", color='#6b7280', fontsize=9)
        ax.set_ylabel("Actual",    color='#6b7280', fontsize=9)
        ax.tick_params(colors='#6b7280', labelsize=8)
        ax.set_facecolor('#ffffff')
        fig.patch.set_facecolor('#ffffff')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📋 Classification Report")
        rows = []
        for sp in iris_d.target_names:
            rows.append({
                "Species":   f"{SPECIES_EMOJI.get(sp,'🌸')} {sp.capitalize()}",
                "Precision": f"{report[sp]['precision']:.2f}",
                "Recall":    f"{report[sp]['recall']:.2f}",
                "F1-Score":  f"{report[sp]['f1-score']:.2f}",
                "Support":   int(report[sp]['support']),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("#### 🐍 Code")
        st.code("""from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report
)

y_pred = model.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2%}")
print(classification_report(y_test, y_pred))
""", language="python")

    st.markdown(f"<div class='success-box'>✅ Ang model ay may <b>{acc*100:.1f}% accuracy</b> sa test set!</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "🎯  Step 5: Predict":
    st.markdown("""
    <div class='hero'>
        <div class='step-pill'>STEP 5</div>
        <div class='hero-title' style='font-size:2.2rem;'>Predict Species</div>
        <div class='hero-desc'>I-input ang flower measurements → i-classify ang iris species</div>
    </div>
    """, unsafe_allow_html=True)

    model, scaler = load_model()
    if model is None:
        st.markdown("<div class='warning-box'>⚠️ Wala pang trained model! Pumunta sa <b>Step 3: Train</b> muna.</div>", unsafe_allow_html=True)
        st.stop()

    st.markdown("### 📏 I-input ang Flower Measurements")

    c1, c2 = st.columns(2)
    with c1:
        sepal_l = st.slider("🌿 Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
        sepal_w = st.slider("🌿 Sepal Width  (cm)", 2.0, 4.5, 3.5, 0.1)
    with c2:
        petal_l = st.slider("🌸 Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
        petal_w = st.slider("🌸 Petal Width  (cm)", 0.1, 2.5, 0.2, 0.1)

    if st.button("🔮 Classify Flower"):
        features   = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
        feat_sc    = scaler.transform(features)
        pred_idx   = model.predict(feat_sc)[0]
        pred_prob  = model.predict_proba(feat_sc)[0]
        pred_name  = load_iris().target_names[pred_idx]
        confidence = pred_prob[pred_idx] * 100
        emoji      = SPECIES_EMOJI.get(pred_name, "🌸")

        st.markdown(f"""
        <div class='pred-box'>
            <div class='pred-emoji'>{emoji}</div>
            <div class='pred-species'>Iris {pred_name.capitalize()}</div>
            <div class='pred-conf'>Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Class Probabilities")

        prob_colors = ["#2d6a4f", "#52b788", "#e07a5f"]
        for sp, prob, color in zip(load_iris().target_names, pred_prob, prob_colors):
            st.markdown(f"""
            <div style='margin-bottom:0.7rem;'>
                <div style='display:flex;justify-content:space-between;
                            color:#1a1a2e;font-size:0.88rem;
                            font-weight:600;margin-bottom:0.3rem;'>
                    <span>{SPECIES_EMOJI.get(sp,'🌸')} Iris {sp.capitalize()}</span>
                    <span style='color:{color};'>{prob*100:.1f}%</span>
                </div>
                <div style='background:#e8e0d5;border-radius:50px;height:10px;'>
                    <div style='background:{color};width:{prob*100:.1f}%;
                                height:10px;border-radius:50px;
                                transition:width 0.5s ease;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        input_df = pd.DataFrame({
            "Feature":    ["Sepal Length","Sepal Width","Petal Length","Petal Width"],
            "Value (cm)": [sepal_l, sepal_w, petal_l, petal_w],
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)

        st.code(f"""model  = joblib.load('iris_knn_model.joblib')
scaler = joblib.load('iris_scaler.joblib')

features = np.array([[{sepal_l}, {sepal_w}, {petal_l}, {petal_w}]])
scaled   = scaler.transform(features)
pred     = model.predict(scaled)

species = ['setosa', 'versicolor', 'virginica']
print(f"Predicted: {{species[pred[0]]}}")
""", language="python")
