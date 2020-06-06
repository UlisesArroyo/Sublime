import cv2
import torch
from torchvision import transforms
from PIL import Image 

transform = transforms.Compose([
	transforms.ToTensor()])

transform_2 =transforms.Compose([
	transforms.ToPILImage(mode = RGB)])




# Cargamos la imagen del disco duro
#imagen = cv2.imread("dog.jpg")
#cv2.imshow("prueba", imagen)
#cv2.waitKey(0)

imagen = Image.open("dog.jpg")
imagen_t = transform(imagen)
imagen_t.cuda
imagen_n = transform(imagen_t)
cv2.imshow("prueba 2",imagen_n)
cv2.waitKey(0)
#img_t = torch.device('cuda',0)
#img_gpu = img_t.to(device = 'cuda')

#img_gpu = torch.tensor(img_t, device = 'cuda')
