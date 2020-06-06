
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable



transform = transforms.Compose([
	 transforms.ToTensor()])

transformacion_2 =transforms.Compose([
	transforms.ToPILImage("RGB")
	])


transform3 = transforms.Compose([
 #  transforms.RandomHorizontalFlip(),
   transforms.ToTensor(),
 # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#   transforms.RandomErasing(),
 ])



 # Cargamos la imagen del disco duro
imagen = cv2.imread("dog.jpg")
 #cv2.imshow("prueba", imagen)
 #cv2.waitKey(0)



 #img = Image.open("dog.jpg") # Por lo que entiendo esta sentencia viene de la libreria de python (PIL) y sirve para 
imagen_t = transform(imagen)
imagen_i = transformacion_2(imagen_t)
width, height = imagen_i.size
imagen_m = imagen_i.crop((0,0,(width/2),height))

imagen_f = transform3(imagen_m)

#imagen_t.cuda
#imagen_nueva = imagen_t[0].numpy().transpose(1,2,0)



imagen_nueva = imagen_f.permute(1,2,0).numpy()
imagen_nueva = (imagen_nueva * 255).round().astype(np.uint8)
cv2.imwrite('nuevo_perro2.jpg', imagen_nueva)
cv2.imshow("prueba_2",imagen_nueva)
cv2.waitKey(0)