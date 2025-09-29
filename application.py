from flask import Flask, request, render_template
import numpy as np
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictions', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html', prediction_text=f'Predicted Math Score: {np.round(prediction, 2)}')

    else:
        try:
            # Validate inputs
            def get_required_field(name):
                val = request.form.get(name)
                if not val:
                    raise ValueError(f"Missing input: {name}")
                return val

            data = CustomData(
                gender=get_required_field('gender'),
                race_ethnicity=get_required_field('race_ethnicity'),
                parental_level_of_education=get_required_field('parental_level_of_education'),
                lunch=get_required_field('lunch'),
                test_preparation_course=get_required_field('test_preparation_course'),
                writing_score=float(get_required_field('writing_score')),
                reading_score=float(get_required_field('reading_score')),
            )

            features_df = data.get_data_as_data_frame()
            print("Final DataFrame for Prediction:")
            print(features_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(features_df)
            prediction = results[0]

            return render_template('home.html', prediction_text=f'Predicted Math Score: {np.round(prediction, 2)}')

        except Exception as e:
            return f"Error: {e}"

