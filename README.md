# Student Performance Predictor

## Overview
This project predicts students' math scores using demographic and academic data such as gender, race/ethnicity, parental education, lunch type, test preparation course status, reading score, and writing score. The prediction is done using a Linear Regression model.

## Dataset
The dataset used is from Kaggle’s "Students Performance in Exams," containing 1000 records with no missing values.

## Features Used
- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch
- Test Preparation Course
- Reading Score
- Writing Score

## Methodology
- Loaded and cleaned the data.
- Encoded categorical features using Label Encoding.
- Used reading and writing scores along with encoded categorical features as input.
- Split data into training (80%) and testing (20%) sets.
- Trained a Linear Regression model.
- Evaluated model performance using Mean Squared Error and R² score.

## Results
- Mean Squared Error: ~36.79
- R² Score: ~0.85

## How to Run
1. Clone the repo:
