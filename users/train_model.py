import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import io
import base64
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor   # <-- ADDED

def train_and_evaluate_models(csv_file='insurance.csv'):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        return {"error": "insurance.csv not found. Please ensure the file is in the correct directory."}
    except Exception as e:
        return {"error": f"Error loading the dataset: {e}"}

    df = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']]  
    df.fillna(df.mean(numeric_only=True), inplace=True)

    label_encoders = {}
    for column in ['sex', 'smoker', 'region']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    X = df.drop('charges', axis=1)
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        # "SVM": SVR(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
         "Decision Tree": DecisionTreeRegressor() 
    }

    results = {}
    prediction_plots = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = {"r2": r2, "mae": mae}

        # Generate prediction plot
        plt.figure(figsize=(6, 4))
        plt.plot(y_test.values, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.title(f'{name} - Actual vs Predicted')
        plt.legend()
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        prediction_plots[name] = plot_url

    # Model comparison bar plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(results.keys()), y=[results[m]["r2"] for m in results])
    plt.ylabel("R² Score")
    plt.title("Model Comparison (R² Score)")
    plt.ylim(0, 1)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    comparison_plot = base64.b64encode(img.getvalue()).decode()

    best_model_name = max(results, key=lambda k: results[k]['r2'])
    best_model = models[best_model_name]
    joblib.dump(best_model, 'best_model.joblib')

    # Save label encoders
    joblib.dump(label_encoders, 'label_encoders.joblib')

    return {
        "results": results,
        "prediction_plots": prediction_plots,
        "comparison_plot": comparison_plot,
        "best_model": best_model_name,
        "message": "All models trained and evaluated successfully."
    }
