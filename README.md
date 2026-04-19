# 🌸 Iris ML Classifier — Batang Quiapo

A step-by-step Machine Learning web app using the **Iris dataset**, built with **Streamlit**, **scikit-learn**, and **joblib**.

---

## 👥 Group: Batang Quiapo

| # | Name | Role |
|---|------|------|
| 1 | John Loise Ibeng | ML Engineer |
| 2 | Jaby Maverick Lasquite | Data Analyst |
| 3 | Rodel Lobendino | Backend Dev |
| 4 | Reymark Morales | UI/UX Designer |
| 5 | Jomel Onido | Project Lead |

---

## 🗺️ ML Pipeline (5 Steps)

| Step | Description |
|------|-------------|
| 📊 Step 1 | Load the Iris dataset from scikit-learn |
| 🔍 Step 2 | Exploratory Data Analysis (box plots, heatmap, scatter) |
| ⚙️ Step 3 | Split, Scale, Train KNN, Save with joblib |
| 📈 Step 4 | Evaluate — confusion matrix, accuracy, classification report |
| 🎯 Step 5 | Predict iris species from user input |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/iris-ml-batang-quiapo.git
cd iris-ml-batang-quiapo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → `app.py` → Deploy!

---

## 🛠️ Tech Stack

- **Streamlit** — Web UI framework
- **scikit-learn** — KNN classifier & preprocessing
- **joblib** — Model persistence
- **pandas / numpy** — Data manipulation
- **matplotlib / seaborn** — Visualizations

---

## 📁 Project Structure

```
iris-ml-batang-quiapo/
├── app.py               # Main Streamlit app
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── iris_knn_model.joblib  # Saved model (after training)
└── iris_scaler.joblib     # Saved scaler (after training)
```

---

> Made with ❤️ by **Batang Quiapo** — Machine Learning Project
