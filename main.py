from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Chargement du modèle Random Forest
rf_model = joblib.load("models/credit_default_model_pipeline.pkl")


# Fonctions de transformation sécurisée
def to_float(value, field_name):
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"Champ '{field_name}' doit être un nombre décimal.")


def to_int(value, field_name):
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Champ '{field_name}' doit être un entier.")


# Fonctions simulées pour NLP et LLM
def fake_nlp_predict(text):
    return f"Analyse NLP: Texte reçu avec {len(text)} caractères."


def fake_llm_predict(text):
    return "Réponse LLM: Question comprise."


# Route principale (GET + POST)
@app.route("/", methods=["GET", "POST"])
def predict_rf():
    result_message = None

    if request.method == "POST":
        try:
            data = request.form

            # --- Construction du dictionnaire de données ---
            input_dict = {
                "limit_bal": to_float(data["limit_bal"], "limit_bal"),
                "sex": data["sex"],
                "education": data["education"],
                "marriage": data["marriage"],
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

            # --- Création du DataFrame ---
            input_df = pd.DataFrame([input_dict])

            # --- Prédiction ---
            prediction = rf_model.predict(input_df)[0]
            probability = rf_model.predict_proba(input_df)[0][1]

            # --- Affichage du résultat ---
            if prediction == 1:
                result_message = f"⚠️ En défaut de paiement avec une probabilité de {round(probability * 100, 2)} %"
            else:
                result_message = f"✅ Pas en défaut de paiement avec une probabilité de {round((1 - probability) * 100, 2)} %"

        except Exception as e:
            result_message = f"Erreur : {str(e)}"

    return render_template("index.html", result_message=result_message)


# Route NLP
@app.route("/predict_nlp", methods=["POST"])
def predict_nlp():
    if request.method == "POST":
        text = request.form["text_nlp"]
        result = fake_nlp_predict(text)
        return render_template("index.html", prediction_nlp=result)


# Route LLM
@app.route("/predict_llm", methods=["POST"])
def predict_llm():
    if request.method == "POST":
        text = request.form["text_llm"]
        result = fake_llm_predict(text)
        return render_template("index.html", prediction_llm=result)


if __name__ == "__main__":
    app.run(debug=True)
