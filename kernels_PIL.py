from PIL import Image, ImageChops, ImageFilter
import numpy as np

img=Image.open('donut.png')
img2=img.convert('L') # podemos poner '1' y 'P'
img3=img.convert('P')
imgmatrix=np.asarray(img) # lo convertimos en array los valores de la imagen
imgmatrix2=np.asarray(img3)#,dtype=np.uint8

# Desenfoque gaussiano 3x3:
kernelValues_1=[1/16,2/16,1/16,2/16,4/16,2/16,1/16,2/16,1/16]
kernel_1=ImageFilter.Kernel((3,3), kernelValues_1)
imagen_1=img2.filter(kernel_1)
#imagen_1.save('Desenfoque_gaussiano3x3.png')

# Enfoque gaussiano 3x3:
kernelValues_2=[-1/16,-2/16,-1/16,-2/16,28/16,-2/16,-1/16,-2/16,-1/16]
kernel_2=ImageFilter.Kernel((3,3), kernelValues_2)
imagen_2=img2.filter(kernel_2)
#imagen_2.save('Enfoque_gaussiano3x3.png')

# Desenfoque gaussiano 5x5:
kernelValues1=[1/256,4/256,6/256,4/256,1/256,4/256,16/256,24/256,16/256,4/256,6/256,24/256,36/256,24/256,6/256
,4/256,16/256,24/256,16/256,4/256,1/256,4/256,6/256,4/256,1/256]
kernel1=ImageFilter.Kernel((5,5), kernelValues1)
imagen1=img2.filter(kernel1)
#imagen1.save('Desenfoque_gaussiano5x5.png')

# Enfoque gaussiano 5x5:
kernelValues2=[-1/256,-4/256,-6/256,-4/256,-1/256,-4/256,-16/256,-24/256,-16/256,-4/256,-6/256,-24/256,476/256,-24/256,-6/256
,-4/256,-16/256,-24/256,-16/256,-4/256,-1/256,-4/256,-6/256,-4/256,-1/256]
kernel2=ImageFilter.Kernel((5,5), kernelValues2)
imagen2=img2.filter(kernel2); imagen2_2=img2.filter(kernel2)
#imagen2.save('Enfoque_gaussiano5x5.png')#; imagen2_2.save('Enfoque_gaussiano5x5prueba.png')

# Difuminado:
kernelValues_3=[1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9]
kernel_3=ImageFilter.Kernel((3,3), kernelValues_3)
imagen_3=img2.filter(kernel_3)
#imagen_3.save('Difuminado.png')

# Enfoque:
kernelValues3=[-1/9,-1/9,-1/9,-1/9,1,-1/9,-1/9,-1/9,-1/9]
kernel3=ImageFilter.Kernel((3,3), kernelValues3)
imagen3=img2.filter(kernel3)
#imagen3.save('Enfoque.png')
#imagen3.show()

# Deteccion de bordes:
kernelValuesx=(-1,-1,-1,-1,8,-1,-1,-1,-1)
kernelx=ImageFilter.Kernel((3,3), kernelValuesx, 1,0) # 1 es el valor por el que se divide a la suma de los
# valores del kernel. La imagen no existira porque -1+...+8=0 y estaria siendo dividida entre 0.
imagenx=img2.filter(kernelx)
#imagenx.save('Bordes.png')
#imagenx.show()

# Sobel:
kernelValuesy=(-1,0,1,-2,0,2,-1,0,1)
kernelValuesz=[-1,-2,-1,0,0,0,1,2,1]
kernelA=ImageFilter.Kernel((3,3), kernelValuesy, 1,0)
kernelB=ImageFilter.Kernel((3,3), kernelValuesz, 1,0)
# Mejor usar .cv2 para aplicar este filtro(s)

