from flask import Flask, render_template, request
import numpy as np
import joblib

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

        # Mapping text to int
        sex_map = {"Male": 0, "Female": 1}
        education_map = {
            "Graduate school": 1,
            "University": 2,
            "High school": 3,
            "Others": 4,
        }
        marriage_map = {"Married": 1, "Single": 2, "Others": 3}

        # Construction du vecteur de features
        features = [
            float(data["limit_bal"]),
            sex_map[data["sex"]],
            education_map[data["education"]],
            marriage_map[data["marriage"]],
            int(data["age"]),
            int(data["payment_status_sep"]),
            int(data["payment_status_aug"]),
            int(data["payment_status_jul"]),
            int(data["payment_status_jun"]),
            int(data["payment_status_may"]),
            int(data["payment_status_apr"]),
            float(data["bill_statement_sep"]),
            float(data["bill_statement_aug"]),
            float(data["bill_statement_jul"]),
            float(data["bill_statement_jun"]),
            float(data["bill_statement_may"]),
            float(data["bill_statement_apr"]),
            float(data["previous_payment_sep"]),
            float(data["previous_payment_aug"]),
            float(data["previous_payment_jul"]),
            float(data["previous_payment_jun"]),
            float(data["previous_payment_may"]),
            float(data["previous_payment_apr"]),
        ]

        # Prédiction
        prediction = rf_model.predict([features])[0]

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
