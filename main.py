from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd


app = Flask(__name__)

# Chargement des modèles (assure-toi que les fichiers existent dans 'models/')
rf_model = joblib.load("models/credit_default_model_pipeline.pkl")


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

        sex_map = {"Male": 0, "Female": 1}
        education_map = {
            "Graduate school": 1,
            "University": 2,
            "High school": 3,
            "Others": 4,
        }
        marriage_map = {"Married": 1, "Single": 2, "Divorced": 3}

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

        input_dict = {
            "limit_bal": to_float(data["limit_bal"], "limit_bal"),
            "sex": sex_map.get(data["sex"], None),
            "education": education_map.get(data["education"], None),
            "marriage": marriage_map.get(data["marriage"], None),
            "age": to_int(data["age"], "age"),
        }

        if None in [input_dict["sex"], input_dict["education"], input_dict["marriage"]]:
            raise ValueError("Valeur invalide pour sex, education ou marriage.")

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

        prediction = rf_model.predict(input_df)[0]

        return render_template("index.html", rf_prediction=prediction)

    except Exception as e:
        return render_template(
            "index.html", rf_prediction=f"Erreur lors de la prédiction RF : {str(e)}"
        )


@app.route("/predict_nlp", methods=["POST"])
def predict_nlp():
    if request.method == "POST":
        text = request.form["text_nlp"]
        result = fake_nlp_predict(text)
        return render_template("index.html", prediction_nlp=result)


@app.route("/predict_llm", methods=["POST"])
def predict_llm():
    if request.method == "POST":
        text = request.form["text_llm"]
        result = fake_llm_predict(text)
        return render_template("index.html", prediction_llm=result)


if __name__ == "__main__":
    app.run(debug=True)
