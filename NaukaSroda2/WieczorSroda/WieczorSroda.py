import numpy as np
from matplotlib import pyplot as plt

def zad1Marcin(dt,T,k,u0,target,histereza):
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
        y[i+1] = u*dt - 3*y[i]*dt + y[i]
        us[i] = u
        e[i] = target - y[i]
    return e,us,t,y

#dt = 0.01
#T = 10
#k = 1
#u0 = 10
#target = 3
#histereza = 0.1

#e,us = zad1Marcin(dt,T,k,u0,target,histereza)
#plt.plot(e)
#plt.show()

def zad2Marcin(dt,T,k,u0,u1,u2,target,histereza1,histereza2):
    t = np.arange(0,T,dt)
    y = np.zeros(len(t)+1)
    x = np.zeros(len(t)+1)
    u = u0
    e = np.zeros(len(t))
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
        x[i+1] = (u - x[i] -2*y[i])*dt/3 + x[i]
        y[i+1] = x[i]*dt + y[i]
        us[i] = u
        e[i] = target - y[i]
    return e,us,y,t

dt = 0.01
T = 50
k = 1
target = 3
u0 = 4
u1 = -20
u2 = 20
histereza1 = 0.1
histereza2 = 0.2

#e,us,y = zad2Marcin(dt,T,k,u0,u1,u2,target,histereza1,histereza2)
#plt.plot(e,label="e")
#plt.plot(y,label="y")
#plt.plot(us)
#plt.legend()
#plt.show()


def zad1Mateusz(dt,T,k,u0,target,histereza):
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
        y[i+1] = (u - y[i])*dt/2 + y[i]
        us[i] = u
        e[i] = target - y[i]
    return e,us

#e,us = zad1Mateusz(dt,T,k,u0,target,0)
#plt.plot(e)
#plt.show()

def zad2Mateusz(dt,T,k,target,u0,u1,u2,histereza1,histereza2):
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
        x[i+1] = (u-x[i]-y[i])*dt/2 + x[i]
        y[i+1] = x[i]*dt+y[i]
        e[i] = target-y[i]
        us[i] = u
    return y,e,us,t

dt = 0.01
T = 50
k = 1
target = 5
u0 = 14
u1 = -20
u2 = 20
histereza1 = 0.2
histereza2 = 0.2
#y,e,us,t = zad2Mateusz(dt,T,k,target,u0,u1,u2,histereza1,histereza2)
#plt.plot(t,y[:-1])
#plt.plot(t,us)
#plt.show()
#e,us,y,t = zad2Marcin(dt,T,k,u0,u1,u2,target,histereza1,histereza2)
#plt.plot(t,y[:-1],label="y(t)")
#plt.plot(t,us)
#plt.legend()
#plt.show()

def zadxMarcin(dt,T,k,u0,u1,u2,target,histereza1,histereza2):
    t = np.arange(0,T,dt)
    y = np.zeros(len(t)+1)
    x = np.zeros(len(t)+1)
    u = u0
    e = np.zeros(len(t))
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
        #x[i+1] = (u - x[i] -2*y[i])*dt/3 + x[i]
        y[i+1] = u*dt -3*y[i]*dt + y[i]
        us[i] = u
        e[i] = target - y[i]
    return e,us,y,t

#dt = 0.01
#T = 50
#k = 1
#u0 = 8
#u1 = -20
#u2 = 20
#target = 2.558
#histereza1 = 0.1
#histereza2 = 0.2
#e,us,y,t = zadxMarcin(dt,T,k,u0,u1,u2,target,histereza1,histereza2)
#plt.plot(t,y[:-1])
#plt.plot(t,us)
#plt.show()

def zadanie1Marcina(dt,T,k,u0,target,histereza):
    t = np.arange(0,T,dt)
    y = np.zeros(len(t)+1)
    u = u0
    e = np.zeros(len(t))
    us = np.zeros(len(t))
    for i in range(len(t)):
        if y[i-1]<y[i] and y[i]<target+histereza:
            u = u0
        if y[i-1]<y[i] and y[i]>target+histereza:
            u = 0
        if y[i-1]>y[i] and y[i]>target - histereza:
            u = 0
        if y[i-1]>y[i] and y[i]<target - histereza:
            u = u0
        y[i+1]=(u-3*y[i])*dt + y[i]
        e[i] = target - y[i]
        us[i] = u
    return y,e,u,t

dt = 0.01
T = 50
k = 1
u0 = 15.5
target = 5
histereza = 0.1
#y,e,u,t = zadanie1Marcina(dt,T,k,u0,target,histereza)
#plt.plot(t,e)
#plt.plot(u)
#plt.show()


def zadanie2Marcina(dt,T,k,target,u0,u1,u2,histereza1,histereza2):
    t = np.arange(0,T,dt)
    y = np.zeros(len(t)+1)
    x = np.zeros(len(t)+1)
    u = u0
    e = np.zeros(len(t))
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
        x[i+1] = (u - x[i] - 2*y[i])*dt/3 + x[i]
        y[i+1] = x[i]*dt + y[i]
        e[i] = target - y[i]
        us[i] = u
    return y,e,us,t

dt = 0.01
T = 50
k = 1
target = 5
u0 = 8
u1 = -20
u2 = 20
histereza1 = 0.1
histereza2 = 0.2
#y,e,us,t = zadanie2Marcina(dt,T,k,target,u0,u1,u2,histereza1,histereza2)
#plt.plot(t,y[:-1])
#plt.plot(t,us)
#plt.show()

def zad1Piotr(dt,T,target,u0,histereza):
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
        y[i+1] = (u - 3*y[i])*dt/2 + y[i]
        us[i] = u
        e[i] = target-y[i]
    return y,t,us,e


#dt = 0.01
#T = 50
#target = 5
#u0 = 17
#histereza = 0.1
#y,t,us,e = zad1Piotr(dt,T,target,u0,histereza)
#plt.plot(t,e)
#plt.show()

def zad2Piotr(dt,T,target,u0,histereza):
    t = np.arange(0,T,dt)
    y = np.zeros(len(t)+1)
    x = np.zeros(len(t)+1)
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
        x[i+1] = u*dt/2 - y[i]*dt - 2*x[i]*dt + x[i]
        y[i+1] = x[i]*dt + y[i]
        us[i] = u
        e[i] = target - y[i]
    return t,e,us

#dt = 0.01
#T = 50
#target = 5
#u0 = 20
#histereza = 0.4
#t,e,us = zad2Piotr(dt,T,target,u0,histereza)
#plt.plot(t,e)
#plt.plot(t,us)
#plt.show()

def trojstanowyZHistereza(dt,T,target,u0,u1,u2,histereza1,histereza2):
    t = np.arange(0,T,dt)
    y = np.zeros(len(t)+1)
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
        y[i+1] = k*u*dt - y[i]*dt + y[i]
        us[i] = u
        e[i] = target - y[i]
    return t,e,us

dt = 0.01
T = 50
target = 3
u0 = 15
u1 = -30
u2 = 30
histereza1 = 0.1
histereza2 = 0.5
t,e,us = trojstanowyZHistereza(dt,T,target,u0,u1,u2,histereza1,histereza2)
plt.plot(t,e)
plt.plot(t,us)
plt.show()