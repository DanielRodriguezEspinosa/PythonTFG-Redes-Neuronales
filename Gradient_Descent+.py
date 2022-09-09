# Gradient Descent: 
import numpy as np; import scipy as sc; import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Funcion:
func=lambda th: np.sin(1/2 *th[0]**2-1/4*th[1]**2 +3)*np.cos(2*th[0]+1-np.e**th[1])
res=100 # resolucion del mapa que estamos generando
_X=np.linspace(-2,2,res)
_Y=np.linspace(-2,2,res); _Z=np.zeros((res,res))
print('_X:',_X); print('_Y:',_Y)
for ix, x in enumerate(_X):
	for iy, y in enumerate(_Y):
		_Z[iy,ix]=func([x,y]) # a la funcion se le pasan los datos como un vector de parametros

plt.contourf(_X,_Y,_Z,res)
plt.colorbar()
# Vamos a generar un punto aleatorio sobre esta superficie:
Theta=np.random.rand(2)*4-2 # rango -2 +2
_T=np.copy(Theta) # copia
plt.plot(Theta[0], Theta[1], 'o', c='white'); print('Theta[0], Theta[1]:',Theta,Theta[1])

# Haremos diferencias finitas para aproximar las derivadas parciales
h=0.001; lr=0.001
gradiente=np.zeros(2)
for _ in range(10000):
	for it, th in enumerate(Theta):
		_T=np.copy(Theta); _T[it]+=h # _T[0], _T[1]
		deriv=(func(_T) - func(Theta))/h
		gradiente[it]=deriv
	Theta-=lr*gradiente # learning rate (lr)
	print(func(Theta))
	if (_ % 100==0):
		plt.plot(Theta[0],Theta[1],'.',c='red') # recorrido de nuestros puntos sobre la superficie
plt.plot(Theta[0], Theta[1], 'o', c='green')
plt.show()

X=np.linspace(-2,2,res)
Y=np.linspace(-2,2,res)
X,Y=np.meshgrid(X,Y)
Z=np.sin(1/2*X**2-1/4*Y**2 +3)*np.cos(2*X+1-np.e**Y)
fig=plt.figure(); ax=fig.gca(projection='3d')
#ax.contour3D(X, Y, Z, res);
figura=ax.plot_surface(X,Y,Z, cmap='viridis',linewidth=0.2, antialiased=True)
ax.grid(True); ax.set_zlim(-1, 1)
ax.xaxis.set_major_locator(LinearLocator(12))
ax.yaxis.set_major_locator(LinearLocator(12))
ax.zaxis.set_major_locator(LinearLocator(5))
fig.colorbar(figura, shrink=0.5, aspect=5) # Add a color bar which maps values to colors.
plt.show()
