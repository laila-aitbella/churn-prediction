from flask import Flask, request, render_template
import pandas as pd
import pickle

from flask import Flask, render_template, Response
import matplotlib.pyplot as plt
import io


app = Flask(__name__)

# Charger le modèle
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analysis')
def analysis():
    return render_template("churn.html")


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        # Récupérer les données du formulaire
        form_data = {
            'TotalCharges': float(request.form['TotalCharges']),
            'tenure': float(request.form['tenure']),
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'Contract_Two year': float(request.form['Contract_Two_year']),
            'PaymentMethod_Electronic check': float(request.form['PaymentMethod_Electronic_check']),
            'InternetService_Fiber optic': float(request.form['InternetService_Fiber_optic']),
            'Contract_One year': float(request.form['Contract_One_year']),
            'OnlineSecurity_Yes': float(request.form['OnlineSecurity_Yes']),
            'PaperlessBilling': float(request.form['PaperlessBilling']),
            'TechSupport_Yes': float(request.form['TechSupport_Yes']),
            'OnlineBackup_Yes': float(request.form['OnlineBackup_Yes']),
            'Partner': float(request.form['Partner']),
            'Dependents': float(request.form['Dependents']),
            'MultipleLines_Yes': float(request.form['MultipleLines_Yes']),
            'StreamingTV_Yes': float(request.form['StreamingTV_Yes']),
            'StreamingMovies_Yes': float(request.form['StreamingMovies_Yes']),
            'DeviceProtection_Yes': float(request.form['DeviceProtection_Yes']),
            'PaymentMethod_Credit card (automatic)': float(request.form['PaymentMethod_Credit_card_automatic']),
            'PaymentMethod_Mailed check': float(request.form['PaymentMethod_Mailed_check']),
            'StreamingTV_No internet service': float(request.form['StreamingTV_No_internet_service'])
        }

        # Convertir en DataFrame
        df = pd.DataFrame([form_data])

        # Assurer l'ordre correct des colonnes
        df = df[['TotalCharges', 'tenure', 'MonthlyCharges', 'Contract_Two year', 'PaymentMethod_Electronic check',
                 'InternetService_Fiber optic', 'Contract_One year', 'OnlineSecurity_Yes', 'PaperlessBilling',
                 'TechSupport_Yes', 'OnlineBackup_Yes', 'Partner', 'Dependents', 'MultipleLines_Yes',
                 'StreamingTV_Yes', 'StreamingMovies_Yes', 'DeviceProtection_Yes', 'PaymentMethod_Credit card (automatic)',
                 'PaymentMethod_Mailed check', 'StreamingTV_No internet service']]

        # Prédire le churn
        prediction = model.predict(df)

        return render_template("prediction.html", prediction_text="La prédiction de churn est {}".format(prediction[0]))        
    else:
        return render_template("prediction.html")
  
  


if __name__ == "__main__":
    app.run(debug=True)
