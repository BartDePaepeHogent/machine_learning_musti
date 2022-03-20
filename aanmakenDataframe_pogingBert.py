from PIL import Image, ImageOps
from numpy import asarray
import numpy as np
import os
import pandas as pd

def aanmakenDf (groep, label):
#Drie lege lijsten worden aangemaakt die dan uiteindelijk samen dataframe zullen vormen 
    Pixels= []
    Label =[]
    Naam=[]
  
#fotos inlezen en omzetten per groep. 
    for foto in os.listdir(groep):
        beeld = Image.open(groep+foto)
#fotos omzetten in grijswaarde of 2 dimensionele array ipv 3x2
        beeld_grijs = ImageOps.grayscale(beeld)

#fotos resizen voor leesbaarheid tijdens testen, en eventueel voor latere experimenten
        beeld_formaat = beeld_grijs.resize((beeld_grijs.width // 32, beeld_grijs.height //32 ))
        
        
#fotos omzetten naar numpy array
        data = asarray(beeld_formaat, dtype=np.uint8)
#3 lijsten opvullen, eerste met data van foto, tweede met label en derde met naam van foto. 
#De lengte van elke lijst = aantal fotos in bijhorende map
        Pixels.append(data)
        Label.append(label)
        Naam.append(foto)

        #Pixels2 = asarray(Pixels)
        Label2 = asarray(Label)
#Per groep worden de drie aangemaakte lijsten omgezet naar panda df
    dict = {'data': Pixels, 'label': Label2, 'naam': Naam}
    df = pd.DataFrame(dict)
    return df

