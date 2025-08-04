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
    if request.method == "POST":
        try:
            features = np.array(
                [
                    [
                        float(request.form["limit_bal"]),
                        int(request.form["sex"]),
                        int(request.form["education"]),
                        int(request.form["marriage"]),
                        int(request.form["age"]),
                        int(request.form["payment_status_sep"]),
                        int(request.form["payment_status_aug"]),
                        int(request.form["payment_status_jul"]),
                        int(request.form["payment_status_jun"]),
                        int(request.form["payment_status_may"]),
                        int(request.form["payment_status_apr"]),
                        float(request.form["bill_statement_sep"]),
                        float(request.form["bill_statement_aug"]),
                        float(request.form["bill_statement_jul"]),
                        float(request.form["bill_statement_jun"]),
                        float(request.form["bill_statement_may"]),
                        float(request.form["bill_statement_apr"]),
                        float(request.form["previous_payment_sep"]),
                        float(request.form["previous_payment_aug"]),
                        float(request.form["previous_payment_jul"]),
                        float(request.form["previous_payment_jun"]),
                        float(request.form["previous_payment_may"]),
                        float(request.form["previous_payment_apr"]),
                    ]
                ]
            )
            prediction = rf_model.predict(features)
            return render_template("index.html", rf_result=prediction[0])
        except Exception as e:
            return f"Erreur lors de la prédiction RF : {e}"


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
