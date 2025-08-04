from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/ton_modele.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # récupération des données du formulaire
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])
    prediction = model.predict(final_features)

    return render_template("index.html", prediction_text=f"Résultat: {prediction[0]}")


if __name__ == "__main__":
    app.run(debug=True)
