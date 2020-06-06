
import cv2
#import torch
#from torchvision import transforms
#from PIL import Image 

#transforms = transforms.Compose([
#	transforms.ToTensor()])

# Cargamos la imagen del disco duro
imagen = cv2.imread("dog.jpg")

cv2.imshow("prueba", imagen)
cv2.waitKey(0)

#img = Image.open("dog.jpg")#Image.open es parte de la biblioteca de imagenes PIL de python
#img_t = transforms(img)
#img_gpu = torch.tensor(img_t, device = 'cpu')
