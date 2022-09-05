import cmath; import math # estas librerías no aceptan listas en tanh(x), etc.
# Usamos numpy para estas funciones
import sympy as sp; import numpy as np
import matplotlib.pyplot as plt; import scipy as sc
#import tensorflow_addons as tfa; import tensorflow as tf

# Step function:
print('Step function:\n')
x=np.linspace(-5,5,100)
print(x)
y=np.zeros(100)#, 9, 16, 25, 49
for i in range(50):
	y[50+i]=1;
plt.step(x, y, where='mid')
plt.show()

# Activation functions:
sigmoide=(lambda x: 1/(1+np.e**(-x)), lambda x: x*(1-x))
relu= lambda x: np.maximum(0,x)
tanh= (lambda x: (np.e**(x)-np.e**(-x))/(np.e**(x)+np.e**(-x)),
	lambda x: 1/(np.cosh(x))**2)

_x=np.linspace(-5,5,100)
plt.plot(_x,sigmoide[0](_x), label='Sigmoide'); plt.plot(_x,sigmoide[1](sigmoide[0](_x)), label='Derivada') 
plt.grid(); plt.legend(); plt.show() # Sigmoide
plt.plot(_x,relu(_x), label='RELU'); plt.grid(); plt.legend(); plt.show() # Relu
plt.plot(_x,tanh[0](_x), label='Tangente Hiperbolica'); plt.plot(_x,tanh[1](_x), label='Derivada') # Tanh
plt.grid(); plt.legend(); plt.show()

# MISH:
def mish(x):
	act_func=x*np.tanh(np.log(1+np.exp(x)))
	return act_func
def mish_prime(z):
	act_func_prime=z*(1-np.tanh(np.log(np.exp(z)+1))**2)*np.exp(z)/(np.exp(z)+1)+np.tanh(np.log(np.exp(z)+1))
	return act_func_prime
z=sp.Symbol('z'); y=z*sp.tanh(sp.log(1+sp.exp(z)))
yprime=sp.diff(y,z,1)
print('Derivative of Mish: ', yprime)

y=mish(x)
yprime=mish_prime(x)
plt.plot(x,y,label='Mish'); plt.plot(x,yprime,label='Derivada')
plt.grid(); plt.legend(); plt.show()

# SOFTMAX:
print('Softmax function:\n')
z=np.linspace(-10,10,100)
z_exp=[math.exp(i) for i in z]
print([round(i, 2) for i in z_exp])
sum_z_exp=sum(z_exp)
softmax=[round(i/sum_z_exp, 3) for i in z_exp]; softmax=np.array(softmax)
plt.plot(z,softmax, label='Softmax')
plt.grid(); plt.legend(); plt.show()

