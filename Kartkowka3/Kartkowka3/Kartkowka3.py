import numpy as np
from matplotlib import pyplot as plt

def zad1(dt,T,target,u0,histereza):
    t = np.arange(0,T,dt)
    y = np.zeros(len(t)+1)
    u = u0
    us = np.zeros(len(t))
    e = np.zeros(len(t))
    for i in range(len(t)):
        if y[i-1]<y[i] and y[i]<target+histereza:
            u = u0
        if y[i-1]<y[i] and y[i]>target+histereza:
            u = 0
        if y[i-1]>y[i] and y[i]>target-histereza:
            u = 0
        if y[i-1]>y[i] and y[i]<target-histereza:
            u = u0
        y[i+1] = (u-3*y[i])*dt + y[i]
        us[i] = u
        e[i] = target - y[i]
    return e,t

#dt = 0.01
#T = 3
#target = 5
#u0 = 16
#histereza = 0.1

#e,t = zad1(dt,T,target,u0,histereza)
#plt.plot(t,e)
#plt.xlabel("t")
#plt.ylabel("E")
#plt.legend()
#plt.show()

def zad2(dt,T,target,u0,u1,u2,histereza1,histereza2):
    t = np.arange(0,T,dt)
    y = np.zeros(len(t)+1)
    x = np.zeros(len(t)+1)
    u = u0
    us = np.zeros(len(t))
    e = np.zeros(len(t))
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
        x[i+1] = (2*u-2*x[i]-y[i])*dt/3+x[i]
        y[i+1] = x[i]*dt +y[i]
        us[i] = u
        e[i] = target - y[i]
    return e,t,us

dt = 0.01
T = 30
target = 5
u0 = 8
u1 = -20
u2 = 20
histereza1 = 0.01
histereza2 = 0.02

e,t,us = zad2(dt,T,target,u0,u1,u2,histereza1,histereza2)
plt.plot(t,e)
plt.plot(t,us)
plt.show()