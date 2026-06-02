# 🏎️ Formula 1 Race Winner Prediction & Performance Analysis

## 📌 Overview

This project focuses on analyzing Formula 1 race data from the 2025 season using the FastF1 library and applying Machine Learning techniques to predict race winners.

The project combines data collection, preprocessing, exploratory data analysis (EDA), feature engineering, machine learning, and interactive visualizations to gain insights into driver and team performance throughout the season.

---

## 🎯 Objectives

- Collect Formula 1 race data using FastF1.
- Perform data cleaning and preprocessing.
- Conduct exploratory data analysis (EDA).
- Visualize driver and constructor performance.
- Engineer new features such as Win and Podium.
- Build a Machine Learning model to predict race winners.
- Deploy insights through an interactive dashboard.

---

## 📊 Dataset Features

| Feature | Description |
|----------|-------------|
| TeamName | Constructor/Team name |
| FullName | Driver name |
| Position | Finishing position |
| Time | Race completion time |
| Status | Race finish status |
| Points | Championship points earned |
| Laps | Total laps completed |
| Venue | Grand Prix location |
| Win | Target variable (1 = Win, 0 = No Win) |
| Podium | Indicates Top 3 finish |

---

## 📈 Exploratory Data Analysis

The project includes several visualizations:

- Driver Championship Standings
- Constructor Championship Standings
- Race Wins by Driver
- Podium Finishes Analysis
- Average Finishing Position
- Points Distribution
- Correlation Heatmap
- Feature Importance Analysis

---

## 🤖 Machine Learning

### Problem Type
Binary Classification

### Target Variable

text
Win
0 = Driver did not win the race
1 = Driver won the race

### Model Used

- Random Forest Classifier

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🛠️ Technologies Used

- Python
- FastF1
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

---

## 📂 Project Structure

text
F1-Win-Prediction/
│
├── F1_Analysis.ipynb
├── f1_2025_results.csv
├── app.py
├── requirements.txt
├── README.md
└── images/

---

## 🚀 Installation

Clone the repository:

bash
git clone https://github.com/yourusername/F1-Win-Prediction.git

Move into the project directory:

bash
cd F1-Win-Prediction

Install dependencies:

bash
pip install -r requirements.txt

## ▶️ Run the Project

Launch the Streamlit dashboard:

bash
streamlit run app.py

## 🔍 Key Insights

- Analysis of Formula 1 2025 race results.
- Identification of top-performing drivers and constructors.
- Visualization of race-winning trends.
- Machine Learning model for race winner prediction.
- Interactive dashboard for data exploration.

---

## 📸 Sample Visualizations

Add screenshots of:

- Driver Standings
- Constructor Standings
- Correlation Heatmap
- Win Distribution
- Streamlit Dashboard

inside the `images` folder.

---

## 📈 Future Improvements

- Include qualifying session data.
- Add lap-by-lap performance analysis.
- Improve feature engineering.
- Experiment with advanced classification models.
- Deploy dashboard publicly.

---

## 👨‍💻 Author

Developed as a Data Analytics and Machine Learning portfolio project using Formula 1 race data.
