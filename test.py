# Mauricio Abarca J.
# 19.319.550-4

import utilities as ut

   
# Feed-forward of the DL
def forward_dl(x,W):        
	# We set-up the weights
	w1 = W[0]
	w2 = W[1]
	w3 = W[2]
	wS = W[3]

	# Activations of AE
	a1, a2 = ut.forward_ae(x, w1, w2)
	a2, a3 = ut.forward_ae(a1, w2, w3)

	# Activation of Softmax
	aS = ut.forward_softmax(a3, wS)

	return aS

# Beginning ...
def main():		
	xv = ut.load_data_csv('test_x.csv')	
	yv = ut.load_data_csv('test_y.csv')	
	W  = ut.load_w_dl()
	zv = forward_dl(xv,W)      		
	ut.metricas(yv,zv) 
	print('Metrica File Generated!')
	

if __name__ == '__main__':   
	 main()