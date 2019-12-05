import numpy as np
from matplotlib import pyplot as plt

def zad2(dt,T,target,k,u0,u1,u2,y2_param,y1_param):
    t = np.arange(0,T,dt)
    y = np.zeros(len(t)+1)
    x = np.zeros(len(t)+1)
    u = u0
    us = np.zeros(len(t))
    for i in range(len(t)):
        pass

def trojstanowyZHisterezaDrugiRzad(dt,T,target,k,u0,u1,u2,histereza1,histereza2,y2_param,y1_param,param):
    t = np.arange(0,T,dt)
    y = np.zeros(len(t)+1)
    x = np.zeros(len(t)+1)
    u = u0
    us = np.zeros(len(t))
    for i in range(len(t)):
        if y[i-1]<y[i]:
            if y[i]<target-histereza1:
                u = u2
            else:
                if y[i]<target+histereza2:
                    u = u0
                else:
                    u = u1
        else:
            if y[i]>target+histereza1:
                u = u1
            else:
                if y[i]>target-histereza2:
                    u = u0
                else:
                    u = u2
        #x[i+1] = (u - 3*x[i] - 2*y[i])*dt/2+x[i]
        x[i+1] = x[i]+dt/(y2_param*y1_param)*(k*u-(y2_param+y1_param)*x[i]-param*y[i])
        #y[i+1] = param*y[i]+dt*x[i]
        y[i+1] = x[i]*dt+y[i]
        us[i] = u
    return target-y,us


k=1
dt=0.01
T = 10
u = 7
target = 6
histereza = 0
#y,us = dwustanowyZHisterezaDrugiRzad(dt,T,u,target,histereza,k)
y,us = trojstanowyZHisterezaDrugiRzad(dt,T,target,k,15,-35,30,0.1,0.1,1,2,2)
#plt.plot(y,label="Bez histerezy")
#plt.plot(us)
y,us = trojstanowyZHisterezaDrugiRzad(dt,T,target,k,15,-35,30,0.1,0.2,1,2,2)
#plt.plot(y,label="Z histereza")
#plt.plot(us,label="his")
#plt.hlines(target,0,1000)
#plt.legend()
#plt.show()

def euler(k,dt,u,y_curr):
    return k*u*dt - y_curr*dt + y_curr

def dwustanowyZHisterezaPierwszyRzad(dt,T,u0,target,histereza,k):
    t = np.arange(0,T,dt)
    y = np.zeros(len(t)+1)
    u = u0
    us = np.zeros(len(t))
    for i in range(len(t)):
        if y[i-1]<y[i] and y[i]<target+histereza:
            u = u0
        if y[i-1]<y[i] and y[i]>target+histereza:
            u = 0
        if y[i-1]>y[i] and y[i]>target-histereza:
            u = 0
        if y[i-1]>y[i] and y[i]<target-histereza:
            u = u0
        y[i+1] = euler(k,dt,u,y[i])
        us[i] = u
    return y,us



y,us = dwustanowyZHisterezaPierwszyRzad(dt,T,u,target,histereza,k)
plt.plot(y)
#plt.plot(us)
plt.show()