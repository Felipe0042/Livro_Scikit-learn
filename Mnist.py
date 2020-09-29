import matplotlib as mpl
import matplotlib.pyplot as plt


from sklearn.datasets import fetch_openml
mnist = fetch_openml(name='mnist_784')
X, y = mnist["data"], mnist["target"]

#print(y[36000])


#salvar figura
import os 
def save_fig(fig_id, tight_layout=True):
    path = os.path.join(fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
    

some_digit = X[36000]
some_digit_image = some_digit.reshape(28,28)

#Tamanho da imagem, tipo de coloração, e interpolacao 
#plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation = "nearest")
#plt.show()


#separar dados entre teste e treinamento
#treinamento primeiras 600000 imagens
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#embaralhar dados de treinamento
import numpy as np 
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#Treinando classificador binario
#detectar o digito 5

y_train_5 = (y_train == '9')  #Verdadeiro para todo digito 5
y_test_5  = (y_test == '9')

#escolher classificor e treina-lo
#Gradiente descendente Estocástico
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)


#testa o conjunto de dados com o y_train_5
#print(sgd_clf.predict([some_digit]))

#teste para avaliar o modelo
#retorna as pontuações de avaliação
from sklearn.model_selection import cross_val_score
#print(cross_val_score(sgd_clf, X_train, y_train_5 , cv=3, scoring="accuracy"))


#construir matriz de confusao
#matriz de confusão retorna as previsoes feitas sendo certas ou erradas

#criando conjunto de previsores para serem comparados 
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

#criando matriz de confusao 
from sklearn.metrics import confusion_matrix
#print(confusion_matrix(y_train_5, y_train_pred))


#precisao e revocação/sensibilidade
from sklearn.metrics import precision_score, recall_score, f1_score
#quantas vezes ele esta certo quando detecta o valor procurado 
precision_score(y_train_5, y_train_pred)
#quantas vezes ele deixa passar o valor procurado
recall_score(y_train_5, y_train_pred)


#media harmonica da precisao e da revocação
print(f1_score(y_train_5, y_train_pred))



#limiar de decisão e compesação de precisão/revocação
#retorna uma pontuacao desse limiar 
y_scores = sgd_clf.decision_function([some_digit])


#aumentando o limiar
#threshold = 2000
#y_some_digit_pred = (y_scores > threshold)
#print(y_some_digit_pred)


#decidindo qual limiar usar
#retorna as pontuacoes 
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method = "decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

#plotando grafico
#precisions - Precisao
#recalls - Revocação
#thresholds - limiar
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precisions")
    plt.plot(thresholds, recalls[:-1], "g-", label="recalls")
    plt.xlabel("thresholds")
    plt.legend(loc="center_left")
    plt.ylim([0,1])
    
    
#plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
#plt.show()

#plt.plot(precisions, recalls)
#plt.show()


#ajustar limiar para 70000

#y_train_pred_90 = (y_scores > 70000)
#precision_score(y_train_5, y_train_pred_90)
#recall_score(y_train_5, y_train_pred_90)

#ajustado para precisao de 90%, contudo a revocacao esta baixa 

#CURVA ROC
from sklearn.metrics import roc_curve


# frp        - Falsos Positivos
# tpr        - Verdaeiros Positivos
# thresholds - Limiar

fpr,tpr,thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
#plot_roc_curve(fpr, tpr)
#plt.show()


#quanto mais distante da curva linear, melhor o resultado da metrica
#a curva ROC é mais usada quando se preocupa mais com falsos negativos do que com falsos positivos
#a curva precisao/revocacao p/r é mais utilizada quando se preocupa mais com falsos positivos do que 
#com falsos negativos.

#calcular area abaixo da curva ROC-AUC para uma instancia 
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

#calcular para todas as instancias
#RandomForestClassifier nao possui o metodo decision_function()
#vai utilizar um metodo predict
#que retorna a probalidade de pertencer em %
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5,cv=3, method = "predict_proba")


#para plotar o ROC precisa de pontuação e nao de probabilidades
y_socres_forest = y_probas_forest[:,1] #pontuacao e a probabilidade da classe positiva (probabilidae de ser 5)
fpr_forest,tpr_forest,thresholds_forest = roc_curve(y_train_5, y_socres_forest)
                                    
                                    
#comparando curvas
plt.plot(fpr,tpr,"b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Floresta Aleatoria")
plt.legend(loc = "lower right")
plt.show()




