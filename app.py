from flask import Flask, request
from flask_restful import Resource, Api

import numpy as np
from keras.models import load_model

from mnist_classification.util import classify

app = Flask(__name__)
api = Api(app)

app.config.from_envvar('FLASK_APP_SETTINGS')

cnn = load_model(app.config['CNN_MODEL_PATH'])


class Classify(Resource):
    def post(self):
        json_data = request.get_json()
        image = np.array(json_data['image'])

        label, probabilities = classify.classify(image, cnn)

        print(label)
        print(probabilities)

        return {
            'label': label,
            'probabilities': probabilities
        }


api.add_resource(Classify, '/mnist')

if __name__ == '__main__':
    app.run(debug=True)
