# Mauricio Abarca J.
# 19.319.550-4
# Aim : Deep-Learning: Training via BP+GD

import utilities as ut
	
# Softmax's training
def train_softmax(x,y,param):
    w     = ut.iniW(...)
    costo = []
    for iter in range(param[0]):        
        # complete code        
    return(w,costo)

# AE's Training 
def train_ae(x,hnode,MaxIter,mu):
    w1,w2 = ut.iniW(...)            
    for iter in range(MaxIter):        
        # complete code...        
    return(w1)
#SAE's Training 
def train_sae(x,param):    
    for hn in range(2,len(param)):
        print('AE={} Hnode={}'.format(hn-1,param[hn]))
        # complete code... 
    return(W,x) 
   
# Beginning ...
def main():
    param_sae, param_sft = ut.load_config()    
    xe              = ut.load_data_csv('train_x.csv')
    ye              = ut.load_data_csv('train_y.csv')
    W,Xr            = train_sae(xe,param_sae) 
    Ws, cost        = train_softmax(Xr,ye,param_sft)
    ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()
