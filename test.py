# Mauricio Abarca J.
# 19.319.550-4

import utilities as ut


# Beginning ...
def main():		
	x, y = ut.load_data_csv('test.csv')		
	W    = ut.load_w_dl()
	z    = ut.forward_dl(x, W)      		
	ut.metricas(y, z) 	
	

if __name__ == '__main__':   
	 main()
