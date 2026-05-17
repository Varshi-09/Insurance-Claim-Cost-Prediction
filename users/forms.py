from django import forms
from .models import UserRegistrationModel


class UserRegistrationForm(forms.ModelForm):
    name = forms.CharField(widget=forms.TextInput(attrs={'pattern': '[a-zA-Z]+'}), required=True, max_length=100)
    loginid = forms.CharField(widget=forms.TextInput(attrs={'pattern': '[a-zA-Z]+'}), required=True, max_length=100)
    password = forms.CharField(widget=forms.PasswordInput(attrs={'pattern': '(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}',
                                                                 'title': 'Must contain at least one number and one uppercase and lowercase letter, and at least 8 or more characters'}),
                               required=True, max_length=100)
    mobile = forms.CharField(widget=forms.TextInput(attrs={'pattern': '[56789][0-9]{9}'}), required=True,
                             max_length=100)
    email = forms.CharField(widget=forms.TextInput(attrs={'pattern': '[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$'}),
                            required=True, max_length=100)
    locality = forms.CharField(widget=forms.TextInput(), required=True, max_length=100)
    address = forms.CharField(widget=forms.Textarea(attrs={'rows': 4, 'cols': 22}), required=True, max_length=250)
    city = forms.CharField(widget=forms.TextInput(
        attrs={'autocomplete': 'off', 'pattern': '[A-Za-z ]+', 'title': 'Enter Characters Only '}), required=True,
        max_length=100)
    state = forms.CharField(widget=forms.TextInput(
        attrs={'autocomplete': 'off', 'pattern': '[A-Za-z ]+', 'title': 'Enter Characters Only '}), required=True,
        max_length=100)
    status = forms.CharField(widget=forms.HiddenInput(), initial='waiting', max_length=100)

    class Meta():
        model = UserRegistrationModel
        fields = '__all__'


class InsuranceForm(forms.Form):
    age = forms.IntegerField(label="Age")
    bmi = forms.FloatField(label="BMI")
    children = forms.IntegerField(label="Children")
    
    sex = forms.ChoiceField(
        choices=[('male', 'Male'), ('female', 'Female')],
        label="Sex"
    )

    smoker = forms.ChoiceField(
        choices=[('yes', 'Yes'), ('no', 'No')],
        label="Smoker"
    )

    region = forms.ChoiceField(
        choices=[
            ('northeast', 'Northeast'),
            ('northwest', 'Northwest'),
            ('southeast', 'Southeast'),
            ('southwest', 'Southwest')
        ],
        label="Region"
    )
def predict_claim(request):

    context = {
        'form': InsuranceForm(),
        'risk_factors': [],
        'recommendations': []
    }

    if request.method == 'POST':
        form = InsuranceForm(request.POST)

        if form.is_valid():

            try:

                model = joblib.load(get_model_path())
                encoders = joblib.load(get_encoder_path())

                data = form.cleaned_data

                # ----------- Force rule for children and smoker ----------- #

                if data['age'] <= 10:
                    data['smoker'] = 'no'
                    data['children'] = 0

                # ----------------------------------------------------------- #

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

                usd_to_inr = 83
                prediction_inr = prediction * usd_to_inr


                # ---------------- Risk Score ---------------- #

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


                # ---------------- Risk Category ---------------- #

                if risk_score <= 3:
                    risk_category = "Low"
                    risk_color = "green"

                elif risk_score <= 6:
                    risk_category = "Moderate"
                    risk_color = "orange"

                else:
                    risk_category = "High"
                    risk_color = "red"


                # ---------------- Risk Factors ---------------- #

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


                # ---------------- Recommendations ---------------- #

                
                    

                context.update({

                    'prediction': f"₹{prediction_inr:,.2f}",
                    'show_result': True,
                    'risk_category': risk_category,
                    'risk_color': risk_color,
                    'risk_factors': risk_factors,
                    'recommendations': recommendations
                })


            except Exception as e:
                context['error'] = f"Prediction error: {str(e)}"

    return render(request, 'users/predictForm.html', context)