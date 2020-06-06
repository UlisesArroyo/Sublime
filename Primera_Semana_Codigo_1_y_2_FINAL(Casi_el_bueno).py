
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable


#1° Tranformacion de nuestro dato numphy a un tensor
transform = transforms.Compose([
	 transforms.ToTensor()])

 # Cargamos la imagen del disco duro
imagen = cv2.imread("dog.jpg")
#ancho,alto,canal = imagen.shape
height,width,canal = imagen.shape

imagen_t = transform(imagen)
imagen_t.cuda


imagen_t = imagen_t[:, 1:1+(height), 1:1+round(width/2)]

imagen_t.cpu


#2° Transformacion de tensor a numphy 
imagen_nueva = imagen_t.permute(1,2,0).numpy()
imagen_nueva = (imagen_nueva * 255).round().astype(np.uint8)



cv2.imwrite('nuevo_medio_perro2.jpg', imagen_nueva)
cv2.imshow("prueba_3",imagen_nueva)
cv2.waitKey(0)






