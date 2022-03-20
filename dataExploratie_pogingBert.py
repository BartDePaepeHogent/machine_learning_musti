import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import SGDClassifier
from aanmakenDataframe_pogingBert import aanmakenDf


#Verschillende Mappen die overeenkomen met verschillende label
#Label 0 = niets
#Label 1 = aanwezig
#Label 2 = buiten

aanwezig = r"classificatie/aanwezig/"  #852 foto's 
buiten = r"classificatie/buiten/"       #389 foto's
niets = r"classificatie/niets/"         #1399 foto's
#Eerst wordt per map een panda dataframe aangemaakt
nietsDf = aanmakenDf(niets, 0)
aanwezigDf = aanmakenDf(aanwezig, 1)
buitenDf = aanmakenDf(buiten, 2)
#vervolgens worden ze alle drie samengevoegd tot 1 groot dataframe
volledigDataframe = pd.concat([nietsDf, aanwezigDf, buitenDf], ignore_index=True)
#Ontdekken volledigDataframe
print('Kolommen in dataframe') 
print(volledigDataframe.keys())
print('Voorbeeld van dataframe')
print(volledigDataframe)
print('Aantal waardes per kollom')
print(volledigDataframe.count())
print('Aantal waardes per categorie')
print(volledigDataframe['label'].value_counts())
print('Controle: 1399+852+389 = ', 1399+852+389)
print('')


#Opsplitsen in gestratificieerde testset en trainingsset
np.random.seed(42)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(volledigDataframe, volledigDataframe['label']):
    strat_train_set = volledigDataframe.loc[train_index]
    strat_test_set = volledigDataframe.loc[test_index] 
#X = data, y = label. Dit voor trainingsset en testset  
X_train, X_test, y_train, y_test = strat_train_set['data'], strat_test_set['data'], strat_train_set['label'], strat_test_set['label']
#Controle van proporties testSet en volledige dataset
print('Proportie per label in testSet')
print(strat_test_set["label"].value_counts() / len(strat_test_set))
print('Proportie per label in volledige dataset')
print(volledigDataframe["label"].value_counts() / len(volledigDataframe)) 
print('')
print('Aantal cases Xtrain, ytrain, Xtest, ytest')
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))
print('')


#Binaire Training adhv Stochastic Gradient Descent
y_train_niets = (y_train == 0)
y_test_niets = (y_test == 0)




sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_niets)



print(sgd_clf.predict([X_test[0]]))
print(y_test[0])



