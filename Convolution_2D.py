import numpy as np; import cv2
import scipy; from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt

img=cv2.imread("F:\\Trabajo de Fin de Grado 2021-2022\\Images\\donut.png",0) # leemos la imagen
#imgray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) # pasamos a grises la imagen

# kernel:
h=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]) # deteccion de bordes
#h=np.array([[1,1,1],[1,1,1],[1,1,1]]) # blur
#h=h/9
"""
# Desenfoque gaussiano 5x5:
h=np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
h=h/256
"""
"""
# Enfoque:
h=np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,-476,24,6],[4,16,24,16,4],[1,4,6,4,1]])
h=h/(-256)
"""
"""
img2=np.zeros(forma) # imagen resultado
# Implementamos el algoritmo de convolucion:
for x in list(range(1, forma[0]-1)):
    for y in list(range(1, forma[1]-1)):
        suma=0
        for i in list(range(-1, 2)):
            for j in list(range(-1, 2)):
                suma=imgray[x-i, y-j]*h[i+1,j+1]+suma
        img2[x,y]=suma

maxs=np.amax(img2)
print('max: ', maxs)
img2=img2*255/maxs
img2=img2.astype(np.uint8)
"""

# convolute with proper kernels
sobelx=cv2.Sobel(img,ddepth=6,dx=1,dy=0,ksize=3)
sobely=cv2.Sobel(img,ddepth=6,dx=0,dy=1,ksize=3) # it seems that cv2.CV_64F is equivalent to ddepth=6
#sobelz=cv2.Sobel(sobelx,ddepth=cv2.CV_16S,dx=0,dy=1,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
plt.subplot(1,2,1); plt.imshow(sobelxx, cmap='gray')
plt.title('Sobel X'); plt.xticks([]); plt.yticks([])
plt.subplot(1,2,2); plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'); plt.xticks([]); plt.yticks([])
plt.show()

"""
sobel, ang=cv2.cartToPolar(sobel_x,sobel_y) # it seems that the depth have to be equal
plt.subplot(1,2,1); plt.imshow(sobelz, cmap='gray')
plt.title('Sobel'); plt.xticks([]); plt.yticks([])
plt.subplot(1,2,2); plt.imshow(sobel, cmap='gray')
plt.title('Sobel'); plt.xticks([]); plt.yticks([])
#plt.imshow(sobel, cmap='gray')
"""
abs_sobelx=cv2.convertScaleAbs(sobelx); abs_sobely=cv2.convertScaleAbs(sobely)
#print(sobelx,abs_sobelx,type(sobelx),type(abs_sobelx))

plt.subplot(1,2,1); plt.imshow(abs_sobelx, cmap='gray')
plt.title('Sobel X'); plt.xticks([]); plt.yticks([])
plt.subplot(1,2,2); plt.imshow(abs_sobely, cmap='gray')
plt.title('Sobel Y'); plt.xticks([]); plt.yticks([])
plt.show()

"""
print(np.linalg.matrix_power(abs_sobelx,2)+np.linalg.matrix_power(abs_sobely,2))
sobelzzz=fractional_matrix_power(np.linalg.matrix_power(abs_sobelx,2)+np.linalg.matrix_power(abs_sobely,2), 0.5)
sobelzz=(sobelx**2+sobely**2)**0.5
sobelz=cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

plt.imshow(sobelz, cmap='gray'); plt.xticks([]); plt.yticks([])
plt.show()
plt.imshow(sobelzz, cmap='gray'); plt.xticks([]); plt.yticks([])
plt.show()
plt.imshow(sobelzzz, cmap='gray'); plt.xticks([]); plt.yticks([])
plt.show()
"""
