
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable


#1° Tranformacion de nuestro dato numphy a un tensor
transform = transforms.Compose([
	 transforms.ToTensor()])


#2° Transformacion ahora de tensor a PILI
transformacion_2 =transforms.Compose([
	transforms.ToPILImage("RGB")
	])


#3° Transformacion de PILI a tensor nuevamente
transform3 = transforms.Compose([
 #  transforms.RandomHorizontalFlip(),
   transforms.ToTensor(),
 ])



 # Cargamos la imagen del disco duro
imagen = cv2.imread("dog.jpg")




 #img = Image.open("dog.jpg") # Por lo que entiendo esta sentencia viene de la libreria de python (PIL) y sirve para 
imagen_t = transform(imagen)
imagen_t.cuda
imagen_t = transformacion_2(imagen_t)
width, height = imagen_t.size
imagen_t = imagen_t.crop((0,0,(width/2),height))

imagen_t = transform3(imagen_t)

imagen_t.cpu


#4° Transformacion de tensor a numphy 
imagen_nueva = imagen_t.permute(1,2,0).numpy()
imagen_nueva = (imagen_nueva * 255).round().astype(np.uint8)



cv2.imwrite('nuevo_medio_perro.jpg', imagen_nueva)
cv2.imshow("prueba_3",imagen_nueva)
cv2.waitKey(0)
