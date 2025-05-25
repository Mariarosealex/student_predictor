import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

def load_data(filepath):
    data = pd.read_csv(filepath)
    print("First 5 rows:\n", data.head())
    print("\nMissing values:\n", data.isnull().sum())
    print("\nSummary statistics:\n", data.describe())
    return data

def preprocess_data(data):
    df = data.copy()
    categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def plot_eda(df):
    plt.figure(figsize=(12,6))
    sns.histplot(df['math score'], kde=True, color='skyblue')
    plt.title('Distribution of Math Scores')
    plt.show()
    
    plt.figure(figsize=(12,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    
    plt.figure(figsize=(12,6))
    sns.boxplot(x='gender', y='math score', data=df)
    plt.title('Math Scores by Gender')
    plt.show()

def train_and_evaluate(X, y):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    # Random Forest
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    # Metrics
    def print_metrics(name, y_test, y_pred):
        print(f"\n{name} Performance:")
        print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        print(f"R2: {r2_score(y_test, y_pred):.2f}")
        
    print_metrics("Linear Regression", y_test, y_pred_lr)
    print_metrics("Random Forest", y_test, y_pred_rf)
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    print(f"\nRandom Forest 5-fold CV R2 scores: {cv_scores}")
    print(f"Mean CV R2: {cv_scores.mean():.3f}")
    
    # Plot predicted vs actual for Random Forest
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred_rf, alpha=0.7, color='green')
    plt.plot([0, 100], [0, 100], 'r--')
    plt.xlabel('Actual Math Scores')
    plt.ylabel('Predicted Math Scores')
    plt.title('Random Forest: Actual vs Predicted Math Scores')
    plt.show()
    
    return rf

def plot_feature_importance(model, X):
    importances = model.feature_importances_
    feature_names = X.columns
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    plt.figure(figsize=(10,6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title('Feature Importance (Random Forest)')
    plt.show()

def save_model(model, filename='student_performance_model.pkl'):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def main():
    data = load_data('StudentsPerformance.csv')
    df, label_encoders = preprocess_data(data)
    plot_eda(df)
    
    # Features and target
    X = df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'reading score', 'writing score']]
    y = df['math score']
    
    model = train_and_evaluate(X, y)
    plot_feature_importance(model, X)
    save_model(model)

if __name__ == "__main__":
    main()
