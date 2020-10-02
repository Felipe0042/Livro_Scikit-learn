#previsao de um numero 
#site kaggle datasets


import pandas as pd
base = pd.read_csv("autos.csv", encoding= 'ISO-8859-1')

# ================== Pré-processamento ==============
#apagar a coluna inteira 
#determinar os previsores
base = base.drop('dateCrawled', axis = 1 )
base = base.drop('dateCreated', axis = 1 )
base = base.drop('nrOfPictures', axis = 1 )
base = base.drop('postalCode', axis = 1 )
base = base.drop('lastSeen', axis = 1 )

#analise da coluna name
#print(base['name'].value_counts())
base = base.drop('name', axis = 1 )

#print(base['seller'].value_counts())
base = base.drop('seller', axis = 1 )

#print(base['offerType'].value_counts())
base = base.drop('offerType', axis = 1 )

#dataframe registro onde preço é menor q 10
i1 = base.loc[base.price <= 10]
#print(base.price.mean())
base = base[base.price>10]

i2 = base.loc[base.price > 350000]
base = base[base.price < 350000]

#verificando se o valor é nulo
(base.loc[pd.isnull(base['vehicleType'])])
(base['vehicleType'].value_counts()) #limousine
(base.loc[pd.isnull(base['gearbox'])])
(base['gearbox'].value_counts())
(base.loc[pd.isnull(base['model'])])
(base['model'].value_counts())
(base.loc[pd.isnull(base['fuelType'])])
(base['fuelType'].value_counts())
(base.loc[pd.isnull(base['notRepairedDamage'])])
(base['notRepairedDamage'].value_counts())

#valores mais comuns para cada campo
valores = {'vehicleType': 'limousine', 'gearbox': 'manuell',
           'model': 'golf', 'fuelType': 'benzin',
           'notRepairedDamage': 'nein' }

#substituir os valores nulos da base de dados pelos selecionados
base = base.fillna(value = valores)


#funcao label-enconder
#todos os previsores
previsores = base.iloc[:, 1:13].values
#valores reais
preco_real = base.iloc[:, 0].values


#transformar dados categoricos(strings) em dados numéricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_previsores = LabelEncoder()
previsores[:,0] = LabelEncoder_previsores.fit_transform(previsores[:,0])
previsores[:,1] = LabelEncoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = LabelEncoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = LabelEncoder_previsores.fit_transform(previsores[:,5])
previsores[:,8] = LabelEncoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = LabelEncoder_previsores.fit_transform(previsores[:,9])
previsores[:,10] = LabelEncoder_previsores.fit_transform(previsores[:,10])

#one hot encoder
#tratar dados categoricos 
# 0 - 0 0 0
# 2 - 0 1 0
# 3 - 0 0 1

onehotencoder = OneHotEncoder(categorical_features = [0,1,3,5,8,9,10])
previsores = onehotencoder.fit_transform(previsores).toarray()



# ================== Processamento ==============







