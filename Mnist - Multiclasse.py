from sklearn.datasets import fetch_openml
import numpy as np 
from sklearn.linear_model import SGDClassifier

mnist = fetch_openml(name='mnist_784')
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


some_digit = X[11] 
print(y[11])
some_digit_image = some_digit.reshape(28,28)

#classificador multiclasse
#utilizando um classificaor binario para tarefa multiclasse

#OVA - Estrátegia um contra todos treina 10 classificador binarios, um para cada digito
#OVO - Estrátegia um contra um, faz a comparaçaõ do digito 0 com 1, 0 com 2, 0 com 3 etc
#ate achar a comparação melhor
#formula N*(N-1)/2

#NORMALMENTE OVA DEMORA MAIS, POIS O CONJUNTO DE DADOS É MUITO GRANDE 

#OVO É MAIS RAPIDO TREINAR VARIOS CLASSIFICADORES EM PEQUENOS CONJUNTOS, DO QUE POUCOS CLASSIFICADORES
#EM GRANDES CONJUNTOS




#utlizar OVA Ou OVO

#OVA
# =============================================================================
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)  #ao inves de y_train_5
sgd_clf.predict([some_digit])
some_digit_scores = sgd_clf.decision_function([some_digit])
# print(some_digit_scores)
# print(np.argmax(some_digit_scores))
# =============================================================================


#OVO
# =============================================================================
# from sklearn.multiclass import OneVsOneClassifier
# ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
# ovo_clf.fit(X_train, y_train)
# print(ovo_clf.predict([some_digit]))
# print(len(ovo_clf.estimators_))
# =============================================================================


#usando classificadores de arvores aleatorios
# =============================================================================
# from sklearn.ensemble import RandomForestClassifier
# forest_clf = RandomForestClassifier(random_state=42)
# forest_clf.fit(X_train, y_train)
# print(forest_clf.predict([some_digit]))
# =============================================================================


#verificando a precisao
# =============================================================================
from sklearn.model_selection import cross_val_predict
#print(cross_val_predict(forest_clf, X_train, y_train, cv=3, method = "predict_proba"))
# =============================================================================


#analisando o erro
#gerando a matriz de confusao
# =============================================================================
# from sklearn.model_selection import cross_val_predict
y_traind_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
# 
# from sklearn.metrics import confusion_matrix
# conf_mt = confusion_matrix(y_train, y_traind_pred)
# =============================================================================
#print(conf_mt)

import matplotlib.pyplot as plt
def mostrar_imagem(nome):
    plt.matshow(nome, cmap=plt.cm.gray)
    plt.show()

#representação em imagem 
#mostrar_imagem(conf_mt)

#dividir o valor da matrizz confusao pelo numero de imagens na classe correspondente

# =============================================================================
# row_sums = conf_mt.sum(axis =1, keepdims = True)
# norm_conf_mx = conf_mt/row_sums
# =============================================================================

#deixar os valores das diagonias com 0, pois so esta medindo o erro
#quanto mais preto = menor o numero de vezes que a previsao aconteceu naquele valor
#as linhas > valores reais
#colunas > previsoes
# =============================================================================
# np.fill_diagonal(norm_conf_mx, 0)
# mostrar_imagem(norm_conf_mx)
# =============================================================================



#========================classificacao Multilabel=============================


# uma instancia com varias classes
# analisa mais de uma coisa por entrada

# =============================================================================
from sklearn.neighbors import KNeighborsClassifier
# 
# y_train_large = (y_train >= '7')
# y_train_odd = (y_train.astype(int) %2 == 1)
# =============================================================================

#concatena os dois vetores
# =============================================================================
# y_multilabel = np.c_[y_train_large, y_train_odd]
# =============================================================================

#KNeighborsClassifier aceita varias classes na instancia
# =============================================================================
knf_clf = KNeighborsClassifier()
#knf_clf.fit(X_train, y_multilabel)
# 
# print(knf_clf.predict([some_digit]))
# 
# 
# from sklearn.model_selection import cross_val_predict
# y_traind_knn_pred = cross_val_predict(knf_clf, X_train, y_multilabel, cv=3)
# =============================================================================


#pontuação F1
#media harmonica da precisao e da revocação
# =============================================================================
# from sklearn.metrics import f1_score
# print(f1_score(y_multilabel,y_traind_knn_pred, average="macro"))
# =============================================================================


#========================classificacao Multioutput ============================
#rotulo(saida) pode ter varios valores

noise = np.random.randint(0,100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0,100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test


some_digit = X_test_mod[500]
some_digit_2 = X_test[500]
some_digit_image = some_digit.reshape(28,28)
some_digit_image_2 = some_digit_2.reshape(28,28)

#Tamanho da imagem, tipo de coloração, e interpolacao 
# =============================================================================
# plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation = "nearest")
mostrar_imagem(some_digit_image)
mostrar_imagem(some_digit_image_2)


some_index = 0

knf_clf.fit(X_train_mod,y_train_mod)
clear_digit = knf_clf.predict([X_test_mod[some_index]])
mostrar_imagem(clear_digit)





