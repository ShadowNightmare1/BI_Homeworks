# Mauricio Abarca J.
# 19.319.550-4

import utilities as ut

   
# Feed-forward of the DL
def forward_dl(x,W):        
	# We set-up the weights
	w1 = W[0]
	w2 = W[1]
	# w3 = W[2]
	wS = W[2]

	# Activations of AE
	a1, a2 = ut.forward_ae(x, w1, w2)
	# a2, a3 = ut.forward_ae(a1, w2, w3)

	# Activation of Softmax
	aS = ut.forward_softmax(a2, wS)

	return aS

# Beginning ...
def main():		
	N = 100
	# xv = ut.load_data_csv('test_x.csv')	
	# yv = ut.load_data_csv('test_y.csv')	
	# llenar a mano x e y
	
	# x = np.array(256, N)
	# y = np.array(256, N)
	
	# for i in range(N):
	# 	x = np.random.rand(0,1)

	# guardar en csv matrices

	# W  = ut.load_w_dl()
	# zv = forward_dl(xv, W)
	# print(zv[:5])      		
	# ut.metricas(yv,zv) 
	

	# print('Metrica File Generated!')
	
	x_test, y_test = ut.generator()
	W  = ut.load_w_dl()
	z_test = forward_dl(x_test, W)
	ut.metricas(y_test, z_test)

	print('Metrica File Generated!')
	

if __name__ == '__main__':   
	 main()