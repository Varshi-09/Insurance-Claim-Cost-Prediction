from ast import alias
from concurrent.futures import process
from django.shortcuts import render

from django.shortcuts import render, HttpResponse, redirect
from django.contrib import messages

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import datetime as dt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import classification_report


def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


import os
import joblib
import pandas as pd
from django.shortcuts import render
from .forms import UserRegistrationForm, InsuranceForm
from django.conf import settings
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from .train_model import train_and_evaluate_models


def get_model_path():
    return os.path.join(settings.BASE_DIR, 'best_model.joblib')


def get_encoder_path():
    return os.path.join(settings.BASE_DIR, 'label_encoders.joblib')


# ✅ ✅ FINAL MODIFIED FUNCTION
def predict_claim(request):

    # ---------------- POST ---------------- #
    if request.method == 'POST':

        form = InsuranceForm(request.POST)

        if form.is_valid():
            try:
                model = joblib.load(get_model_path())
                encoders = joblib.load(get_encoder_path())

                data = form.cleaned_data

                input_df = pd.DataFrame([{
                    'age': data['age'],
                    'sex': data['sex'],
                    'bmi': data['bmi'],
                    'children': data['children'],
                    'smoker': data['smoker'],
                    'region': data['region']
                }])

                for col in ['sex', 'smoker', 'region']:
                    input_df[col] = encoders[col].transform(input_df[col])

                prediction = model.predict(input_df)[0]
                prediction_inr = prediction * 83

                # ---------- Risk Score ----------
                risk_score = 0

                if data['age'] > 60:
                    risk_score += 3
                elif data['age'] > 45:
                    risk_score += 2
                elif data['age'] > 30:
                    risk_score += 1

                if data['bmi'] >= 35:
                    risk_score += 4
                elif data['bmi'] >= 30:
                    risk_score += 3
                elif data['bmi'] >= 25:
                    risk_score += 2

                if data['smoker'] == 'yes':
                    risk_score += 4

                if data['children'] >= 3:
                    risk_score += 1

                if prediction_inr > 20000:
                    risk_score += 3
                elif prediction_inr > 10000:
                    risk_score += 2

                # ---------- Category ----------
                if risk_score <= 3:
                    risk_category = "Low"
                    risk_color = "green"
                elif risk_score <= 6:
                    risk_category = "Moderate"
                    risk_color = "orange"
                else:
                    risk_category = "High"
                    risk_color = "red"

                # ---------- Factors ----------
                risk_factors = []

                if data['smoker'] == 'yes':
                    risk_factors.append("Smoking habit")

                if data['bmi'] >= 30:
                    risk_factors.append("High BMI")
                elif data['bmi'] >= 25:
                    risk_factors.append("Overweight BMI")

                if data['age'] >= 50:
                    risk_factors.append("Age above 50")

                risk_factors = risk_factors[:2]

                # ---------- Recommendations ----------
                recommendations = []

               

                if data['bmi'] >= 35:
                    recommendations.append("Your BMI indicates severe obesity.Maintain a balanced diet and regular exercise routine to reduce BMI and lower health risks.")

                elif data['bmi'] >= 30:
                    recommendations.append("Your BMI suggests obesity.Aim for at least 30 minutes of physical activity daily and diet control.")

                elif data['bmi'] >= 25:
                    recommendations.append("You are slightly overweight. Maintain regular exercise and diet control.")

                if data['smoker'] == 'yes':
                    recommendations.append("Smoking increases hospitalization risk. Consider quitting smoking significantly reduces health risks and insurance costs. .")

                if data['age'] >= 50:
                    recommendations.append(" Maintain a healthy lifestyle and Schedule regular medical check-ups to monitor health conditions.")

                elif data['age'] >= 30:
                    recommendations.append("Maintain an active lifestyle with proper sleep, diet, and physical activity.")

                if data['children'] >= 3:
                    recommendations.append("Consider upgrading family health insurance coverage and long-term financial security.")

                if risk_category == "High":
                    recommendations.append("Maintain a healthy lifestyle, avoid smoking, and get regular health check-ups to reduce future medical risks.")

                if risk_category == "Low":
                    recommendations.append("Continue maintaining a healthy lifestyle and balanced diet.")



                # ✅ SAVE EVERYTHING IN SESSION
                request.session['prediction'] = f"₹{prediction_inr:,.2f}"
                request.session['risk_category'] = risk_category
                request.session['risk_color'] = risk_color
                request.session['risk_factors'] = risk_factors
                request.session['recommendations'] = recommendations

                # ✅ SAVE FORM DATA ALSO
                request.session['form_data'] = request.POST

                # ✅ REDIRECT
                return redirect('predict_claim')

            except Exception as e:
                messages.error(request, f"Prediction error: {str(e)}")

    # ---------------- GET ---------------- #
    form_data = request.session.pop('form_data', None)

    context = {
        'form': InsuranceForm(form_data) if form_data else InsuranceForm(),
        'prediction': request.session.pop('prediction', None),
        'risk_category': request.session.pop('risk_category', None),
        'risk_color': request.session.pop('risk_color', None),
        'risk_factors': request.session.pop('risk_factors', []),
        'recommendations': request.session.pop('recommendations', [])
    }

    return render(request, 'users/predictForm.html', context)


def basefunction(request):
    return render(request, 'base.html')


def training(request):
    results = train_and_evaluate_models()
    models = results["results"]

    best_model = max(models, key=lambda k: models[k]['r2'])
    best_metrics = models[best_model]

    return render(request, 'users/training.html', {
        'results': results,
        "best_model": best_model,
        "best_metrics": best_metrics
    })


def UserLogout(request):
    try:
        del request.session['loggeduser']
    except:
        pass
    return render(request, 'predictor/index.html')


def viewdataset(request):
    return render(request, 'users/viewdataset.html')


def model_detail(request, model_name):

    from .train_model import train_and_evaluate_models

    results = train_and_evaluate_models()

    best_model = results["best_model"]
    best_metrics = results["results"][best_model]

    metrics = results["results"].get(model_name)
    graph = results["prediction_plots"].get(model_name)

    return render(request,'users/model_detail.html',{
        "model": model_name,
        "metrics": metrics,
        "best_model":best_model,
        "graph": graph
    })