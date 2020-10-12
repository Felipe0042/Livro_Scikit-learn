import numpy as np 
b = np.array([[0, 1, 2], [3, 4, 5]]) 


soma= b.sum(axis=-1, keepdims = True)


valor_teste = (b >= 7)

g = [1,2,3,4,5]

noise = np.random.randint(0,100, (len(g), 784))