import os.path

import joblib


class MustiModel:

    def __init__(self):
        self.model = joblib.load(os.path.join("models","musti_model.pkl"))

