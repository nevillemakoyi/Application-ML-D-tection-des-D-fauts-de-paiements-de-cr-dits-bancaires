from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd


app = Flask(__name__)

# Chargement des modèles (assure-toi que les fichiers existent dans 'models/')
rf_model = joblib.load("models/credit_default_model_pipeline.pkl")
# Pour le modèle NLP, on charge le pipeline
nlp_model = joblib.load("models/sentiment_model.pkl")


# Pour les deux autres, tu peux créer des modèles factices ou charger les tiens
# Ici on simule juste les réponses pour l'exemple
def fake_nlp_predict(text):
    # Simule une prédiction pour NLP
    return "Analyse NLP: Texte reçu avec {} caractères.".format(len(text))


def fake_llm_predict(text):
    # Simule une prédiction pour LLM
    return "Réponse LLM: Question comprise."


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_rf", methods=["POST"])
def predict_rf():
    try:
        data = request.form

        def to_float(val, field_name):
            try:
                return float(val)
            except Exception:
                raise ValueError(f"Champ {field_name} doit être un nombre valide.")

        def to_int(val, field_name):
            try:
                return int(val)
            except Exception:
                raise ValueError(f"Champ {field_name} doit être un entier valide.")

        # NE PAS FAIRE DE MAPPING ICI - laisser les valeurs brutes
        input_dict = {
            "limit_bal": to_float(data["limit_bal"], "limit_bal"),
            "sex": data["sex"],  # ex: 'Male'
            "education": data["education"],  # ex: 'Graduate school'
            "marriage": data["marriage"],  # ex: 'Single'
            "age": to_int(data["age"], "age"),
        }

        months = ["sep", "aug", "jul", "jun", "may", "apr"]
        for m in months:
            input_dict[f"payment_status_{m}"] = to_int(
                data[f"payment_status_{m}"], f"payment_status_{m}"
            )
            input_dict[f"bill_statement_{m}"] = to_float(
                data[f"bill_statement_{m}"], f"bill_statement_{m}"
            )
            input_dict[f"previous_payment_{m}"] = to_float(
                data[f"previous_payment_{m}"], f"previous_payment_{m}"
            )

        input_df = pd.DataFrame([input_dict])

        # LAISSE LE PIPELINE FAIRE LA TRANSFORMATION
        prediction = rf_model.predict(input_df)[0]
        # Probabilité associée à la classe prédite
        probability = rf_model.predict_proba(input_df)[0][1]

        # Affichage du message
        if prediction == 1:
            result_message = f"⚠️ En défaut de paiement avec une probabilité de {round(probability * 100, 2)} %"
        else:
            result_message = f"✅ Pas en défaut de paiement avec une probabilité de {round((1 - probability) * 100, 2)} %"

        return render_template("index.html", rf_prediction=result_message)

    except Exception as e:
        return render_template(
            "index.html", rf_prediction=f"Erreur lors de la prédiction RF : {str(e)}"
        )


@app.route("/predict_nlp", methods=["POST"])
def predict_nlp():
    result_nlp = None

    if request.method == "POST":
        text = request.form["text_nlp", ""]
        if text.strip() == "":
            result_nlp = "Veuillez entrer un texte pour l'analyse NLP."

        else:
            # Utilisation du modèle NLP pour prédire le sentiment
            pred = nlp_model.predict([text])[0]
            if pred == 1:
                result_nlp = "Le sentiment est positif."
            else:
                result_nlp = "Le sentiment est négatif."

        return render_template("index.html", prediction_nlp=result_nlp)


@app.route("/predict_llm", methods=["POST"])
def predict_llm():
    if request.method == "POST":
        text = request.form["text_llm"]
        result = fake_llm_predict(text)
        return render_template("index.html", prediction_llm=result)


if __name__ == "__main__":
    app.run(debug=True)
