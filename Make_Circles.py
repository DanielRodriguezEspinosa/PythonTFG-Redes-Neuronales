# Red neuronal: clasificacion de dos grupos
import numpy as np; import scipy as sc
import matplotlib.pyplot as plt
import time; from IPython.display import clear_output
from sklearn.datasets import make_circles
from mpl_toolkits.mplot3d import axes3d
# Creemos el Dataset
n=500 # numero de registros que tenemos en nuestros datos
p=2 # caracteristicas (inputs)
X, Y=make_circles(n_samples=n, factor=0.5, noise=0.05)

Y=Y[:,np.newaxis]
plt.scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1], c='skyblue')
plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1], c='salmon')
plt.axis('equal'); plt.grid(True); plt.show()

# Empecemos a programar nuestra red neuronal:
class neural_layer(): # parametros de la capa: W, b
	def __init__(self, n_conn, n_neur, act_func):
		self.act_func=act_func
		self.b= np.random.rand(1,n_neur)*2-1 # bias(); range->[-1,1]
		self.W=np.random.rand(n_conn,n_neur)*2-1

# FUNCIONES DE ACTIVACION:
sigmoide=(lambda x: 1/(1+np.e**(-x)), lambda x: x*(1-x))
relu= lambda x: np.maximum(0,x)
tanh= (lambda x: (np.e**(x)-np.e**(-x))/(np.e**(x)+np.e**(-x)),
	lambda x: 1/(np.cosh(x))**2)
"""
_x=np.linspace(-5,5,100)
plt.plot(_x,sigmoide[0](_x), label='Sigmoide'); plt.plot(_x,sigmoide[1](sigmoide[0](_x)), label='Derivada') 
plt.grid(); plt.legend(); plt.show() # Sigmoide
plt.plot(_x,relu(_x), label='RELU'); plt.grid(); plt.legend(); plt.show() # Relu
plt.plot(_x,tanh[0](_x), label='Tangente Hiperbolica'); plt.plot(_x,tanh[1](_x), label='Derivada') # Tanh
plt.grid(); plt.legend(); plt.show()
"""
def create_nn(topology,act_func): # creamos la red neuronal
	nn=[] # vector de capas de la red
	for l, layer in enumerate(topology[:-1]):
		nn.append(neural_layer(topology[l], topology[l+1], act_func))
	return nn

topology=[p,4,4,1] # topologia de la red
# (2 inputs, 4 neuronas (fully-connected), 4 neuronas (fully-connected), 1 output)
neural_net=create_nn(topology, sigmoide)
l2_cost=(lambda Yp,Yr: np.mean((Yp-Yr)**2), lambda Yp,Yr: (Yp-Yr))

def train(neural_net,X,Y,l2_cost,lr=0.5,train=True):
	# Forward pass:
	output=[(None, X)]
	for l, layer in enumerate(neural_net):
		z=output[-1][1]@neural_net[l].W + neural_net[l].b 
		a=neural_net[l].act_func[0](z)
		output.append((z,a))
	if train:
		# Backpropagation:
		deltas=[]
		for l in reversed(range(0,len(neural_net))):
			z=output[l+1][0]
			a=output[l+1][1]
			if l==len(neural_net)-1: # Calculando delta de la ultima capa
				deltas.insert(0,l2_cost[1](a,Y)*neural_net[l].act_func[1](a)) # le pasamos la a -> a(1-a)
			else:
				deltas.insert(0,deltas[0]@_W.T*neural_net[l].act_func[1](a))
			_W=neural_net[l].W
			# Gradient Descent: 			
			# optimizando el coste en funcion del parametro de bias:
			neural_net[l].b=neural_net[l].b-np.mean(deltas[0], axis=0, keepdims=True)*lr
			# optimizando el coste en funcion del parametro W:
			neural_net[l].W=neural_net[l].W-output[l][1].T@deltas[0]*lr
	return output[-1][1]

neural_n=create_nn(topology,sigmoide); loss=[]
print("neural_n.b= ",neural_n[0].b,len(neural_n))
for i in range(2500):
	pY=train(neural_n,X,Y,l2_cost,lr=0.01)
	if i%50==49: # 49,99,...,2499 (0-2499).
		loss.append(l2_cost[0](pY,Y))
		res=100; _x0=_x1=np.linspace(-1.5,1.5,res); _Y=np.zeros((res,res))
		for i0, x0 in enumerate(_x0):
			for i1, x1 in enumerate(_x1):
				_Y[i0,i1]=train(neural_n,np.array([[x1,x0]]),Y,l2_cost,train=False)[0][0]
		plt.pcolormesh(_x1,_x0,_Y,cmap='coolwarm'); plt.axis('equal')
		plt.scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1], c='skyblue')
		plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1], c='salmon')
		clear_output(wait=True); plt.show()
		plt.plot(range(len(loss)), loss); plt.show()
		time.sleep(0.5)

fig = plt.figure()
ax = fig.gca(projection='3d')
cset = ax.contourf(_x0, _x1,_Y, cmap='coolwarm')
#ax.clabel(cset, fontsize=9, inline=1)
plt.show()
