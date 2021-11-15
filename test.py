# Mauricio Abarca J.
# 19.319.550-4

import utilities as ut


# Feed-forward of the DL
def forward_dl(x,W):        
	# We set-up the weights
	w1 = W[0]
	w2 = W[1]
	w3 = W[2]
	#w4 = W[3]
	wS = W[3]

	# Activations of AE
	a1 = ut.forward_ae(x, w1)
	a2 = ut.forward_ae(a1, w2)
	a3 = ut.forward_ae(a2, w3)
	#a4 = ut.forward_ae(a3, w4)

	# Activation of Softmax
	aS = ut.forward_softmax(a3, wS) # por alguna razón esta puro fallando cuando es a3

	return aS


# Beginning ...
def main():		
	xv     = ut.load_data_csv('test_x.csv')	
	yv     = ut.load_data_csv('test_y.csv')	
	W      = ut.load_w_dl()
	zv     = forward_dl(xv, W)      		
	# print(zv.shape)
	ut.metricas(yv, zv) 
	print('Metrica File Generated!')
	
	
if __name__ == '__main__':   
	 main()
