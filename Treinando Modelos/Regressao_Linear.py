import numpy as np
import matplotlib.pyplot as plt

x = 2*np.random.rand(100,1)
y = 4 + 3*x + 2*np.random.rand(100,1)

#grafico de dispersão
plt.scatter(x, y)
plt.title('Gráfico de Dispersão entre x e y')
#plt.show()

x_b = np.c_[np.ones((100,1)), x]
#g = np.dot(y)
valor_inicial = x_b.T.dot(x_b)
valor = np.linalg.inv(x_b.T.dot(x_b))
#theta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)