import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tf2ss
import scipy.integrate as integrate
from scipy.optimize import fmin

def PID_correct(A,B,C,D,dt,T,target,k,Ti=None,Td=None):
    t = np.arange(0,T,dt)
    x_od_t = np.zeros((A.shape[0],len(t)+1))
    epsilons = np.zeros((1,len(t)))
    epsilons[0,-1] = target
    epsilons[0,-2] = target
    pole = 0.0
    for i in range(len(t)):
        _P = k * epsilons[0,i-1]
        _I = 0.0
        if Ti is not None:
           _I = k / Ti * pole
        _D = 0.0
        if Td is not None:
            _D = k * Td * (epsilons[0,i-1] - epsilons[0,i-2])/dt
        u = _P + _I + _D
        x_kropka = (A @ x_od_t[:, i].reshape(A.shape[1],1) + B * u).T
        x_od_t[:, i + 1] = x_od_t[:, i] + dt * x_kropka
        y = C @ x_od_t[:, i].reshape(C.shape[1],1) + D * u 
        epsilons[0,i] = target - y
        pole = pole + (epsilons[0,i-1] + epsilons[0,i])*dt/2
    return t,epsilons[0,:]


l = [2.]
m = [10.,0.,3.,1.]
_sys = tf2ss(l,m)
A,B,C,D = _sys
dt = 0.1
T = 100
target = 20




#l = [2.]
#m = [2.,2.,1.]
#_sys = tf2ss(l,m)
#A,B,C,D = _sys
#dt = 0.1
#T = 100
#target = 20

#best_p = fmin(PID_optimize,[1.,1.,1.],args = (A,B,C,D,dt,T,target))
#print(best_p)

#k= 0.000523223275
#Ti= 0.145059717
#Td = 1.80006843
k = 3
Ti = 2
#a,b = PID_correct(A,B,C,D,dt,T,target,k,Ti,0.1)
#c,d = PID_correct(A,B,C,D,dt,T,target,k,Ti,0.2)
#e,f = PID_correct(A,B,C,D,dt,T,target,k,Ti,0.3)
#g,h = PID_correct(A,B,C,D,dt,T,target,k,Ti,0.04)
#plt.plot(a,b,label="Td = 0.1")
#plt.plot(c,d,label="Td = 0.2")
#plt.plot(e,f,label="Td = 0.3")
#plt.plot(g,h,label="Td = 0.4")
#plt.xlabel("t")
#plt.ylabel("E(t)")
#plt.legend()
#plt.show()



def PID_optimize(wektor_poczatkowy,A,B,C,D,dt,T,target):
    k,Ti,Td = wektor_poczatkowy
    x,y = PID_correct(A,B,C,D,dt,T,target,k,Ti,Td)
    return integrate.trapz(y ** 2, x, dt)


#best_p = fmin(PID_optimize,[1.,1.,1.,],args =(A,B,C,D,dt,T,target))
#print(best_p)
l = [2.]
m = [4.,2.,1.]
_sys = tf2ss(l,m)
A,B,C,D = _sys
T = 20
k = 7.39491107
Ti = 6.89248471
Td = 0.47228171
a,b = PID_correct(A,B,C,D,dt,T,target,k,Ti,Td)
plt.plot(a,b)
plt.show()