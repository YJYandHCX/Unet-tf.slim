import numpy as np
import cv2

def resize_image(name):
	read_path = './'+str(name)+'.png'	
	im = cv2.imread(read_path)
	im_re = cv2.resize(im,(300,300))
	save_path = './image/'+str(name)+'.png'
	cv2.imwrite(save_path,im_re)
	return 0
for i in range(1,11):
	_ = resize_image(i)
print ("done")
