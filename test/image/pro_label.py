import cv2
import numpy as np
import pickle

def p_label(name):
	name = str(name)
	p_name = './'+name +'.png'
	im = cv2.imread(p_name)
	
	pp = np.zeros([300,300],dtype = np.int32)
	for i in range(300):
		for o in range(300):
			if im[i,o,2] == 255:
				pp[i,o] = 1
	f=open('./label/'+name+'.pkl','wb')
	pickle.dump(pp,f)
	f.close()
	return 0
	
for i in range(1,11):
	_ = p_label(i)
