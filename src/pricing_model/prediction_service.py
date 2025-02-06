from flask import Flask, request, jsonify, render_template
import pandas as pd
from db_utils import connect_db, load_artifact_from_db
from config import CONFIG

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Expect a JSON payload with the feature values
    data = request.get_json(force=True)
    feature_names = CONFIG["features"]
    
    # Ensure the input is in the proper DataFrame format
    df = pd.DataFrame([data], columns=feature_names)
    
    # Connect to the database and load the scaler and model
    db_path = CONFIG["paths"]["db"]
    conn = connect_db(db_path)
    scaler = load_artifact_from_db(conn, "ridge_scaler.pkl")
    model = load_artifact_from_db(conn, "ridge_model.pkl")
    conn.close()
    
    # Transform the input data and make a prediction
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)
    
    return jsonify({"prediction": float(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
