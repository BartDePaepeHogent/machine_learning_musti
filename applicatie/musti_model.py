import os.path
from PIL import Image, ImageOps
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import joblib


class MustiModel:

    def __init__(self):
        self.model = joblib.load(os.path.join("models","musti_model.pkl"))

    def aanmakenDf(self, groep, label):
        # Drie lege lijsten worden aangemaakt die dan uiteindelijk samen dataframe zullen vormen
        Pixels = []
        Label = []
        # BDP: we hebben naam niet nodig om te trainen, enkel pixels en labels => verwijder
        Naam = []

        # fotos inlezen en omzetten per groep.
        for foto in os.listdir(groep):
            beeld = Image.open(groep + foto)
            # fotos omzetten in grijswaarde of 2 dimensionele array ipv 3x2
            beeld_grijs = ImageOps.grayscale(beeld)

            # fotos resizen voor leesbaarheid tijdens testen, en eventueel voor latere experimenten
            beeld_formaat = beeld_grijs.resize((beeld_grijs.width // 1, beeld_grijs.height // 1))

            # fotos omzetten naar numpy array
            data_2dim = np.asarray(beeld_formaat, dtype=np.uint8)
            data = data_2dim.flatten()

            # 3 lijsten opvullen, eerste met data van foto, tweede met label en derde met naam van foto.
            # De lengte van elke lijst = aantal fotos in bijhorende map
            Pixels.append(data)
            Label.append(label)
            Naam.append(foto)

        # Per groep worden de drie aangemaakte lijsten omgezet naar panda df
        # BDP: we hebben naam niet nodig om te trainen, enkel pixels en labels => verwijder

        dict = {'data': Pixels, 'label': Label, 'naam': Naam}
        df = pd.DataFrame(dict)
        return df

    def train(self):
        # Dataframe aanmaken
        aanwezig = r"classificatie/aanwezig/"  # 852 foto's
        buiten = r"classificatie/buiten/"  # 389 foto's
        niets = r"classificatie/niets/"  # 1399 foto's
        # Eerst wordt per map een panda dataframe aangemaakt
        nietsDf = self.aanmakenDf(niets, 0)
        aanwezigDf = self.aanmakenDf(aanwezig, 1)
        buitenDf = self.aanmakenDf(buiten, 2)
        # vervolgens worden ze alle drie samengevoegd tot 1 groot dataframe
        volledigDataframe = pd.concat([nietsDf, aanwezigDf, buitenDf], ignore_index=True)

        # Opsplitsen in gestratificieerde testset en trainingsset
        # BDP: onderstaande lijn mag weg, seeding zit in StratifiedShuffleSplit
        np.random.seed(42)
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(volledigDataframe, volledigDataframe['label']):
            strat_train_set = volledigDataframe.loc[train_index]
            strat_test_set = volledigDataframe.loc[test_index]
            # X = data, y = label. Dit voor trainingsset en testset
        # BDP: we hebben naam niet nodig om te trainen, enkel pixels en labels => verwijder

        X_train, X_test, y_train, y_test, naam_train, naam_test = strat_train_set['data'], strat_test_set['data'], \
                                                                  strat_train_set['label'], strat_test_set['label'], \
                                                                  strat_train_set['naam'], strat_test_set['naam']
        # Omzetten naar np.array
        X_train_array = np.array(X_train.tolist())
        X_test_array = np.array(X_test.tolist())
        y_train_array = np.array(y_train)
        y_test_array = np.array(y_test)
        # BDP: we hebben naam niet nodig om te trainen, enkel pixels en labels => verwijder

        naam_train_array = np.array(naam_train)
        naam_test_array = np.array(naam_test)

        # scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_array.astype(np.float64))

        self.model.fit(X_train_scaled, y_train_array)
