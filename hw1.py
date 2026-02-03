import numpy as np
from numpy import cos,sin
import matplotlib.pyplot as plt

def ROM_composite_properties(Ef, vf, Em, vm, Vf):

    Vm = 1 - Vf
    
    Gf = Ef / (2*(1+vf))
    Gm = Em / (2*(1+vm))
    
    E1 = Ef*Vf + Em*Vm
    E2 = 1 / (Vf/Ef + Vm/Em) #ROM
    
    G12 = 1 / (Vf/Gf + Vm/Gm) #ROM
    
    nu12 = vf*Vf + vm*Vm
    nu21 = nu12 * (E2/E1)
    
    den = 1 - nu12*nu21
    
    C11 = E1 / den
    C22 = E2 / den
    C12 = nu12 * E2 / den
    C66 = G12
    
    C = np.array([
        [C11, C12, 0],
        [C12, C22, 0],
        [0,   0,   C66]
    ])
    
    S = np.linalg.inv(C)
    
    return C, S

def HalpinTsai_composite_properties(Ef, vf, Em, vm, Vf):

    Vm = 1 - Vf
    
    Gf = Ef / (2*(1+vf))
    Gm = Em / (2*(1+vm))
    
    E1 = Ef*Vf + Em*Vm
    xi_E = 2.0
    eta_E = (Ef/Em - 1) / (Ef/Em + xi_E)
    E2 = Em * (1 + xi_E*eta_E*Vf) / (1 - eta_E*Vf) #Halpin-Tsai
    
    xi_G = 1.0
    eta_G = (Gf/Gm - 1) / (Gf/Gm + xi_G)
    G12 = Gm * (1 + xi_G*eta_G*Vf) / (1 - eta_G*Vf) #Halpin-Tsai
    
    nu12 = vf*Vf + vm*Vm
    nu21 = nu12 * (E2/E1)
    
    den = 1 - nu12*nu21
    
    C11 = E1 / den
    C22 = E2 / den
    C12 = nu12 * E2 / den
    C66 = G12
    
    c = np.array([
        [C11, C12, 0],
        [C12, C22, 0],
        [0,   0,   C66]
    ])
    
    s = np.linalg.inv(c)
    
    return c, s



# Problem 7
C, S = ROM_composite_properties(Ef=430, vf=0.18, Em=71, vm=0.35, Vf=0.42)
c, s= HalpinTsai_composite_properties(Ef=430, vf=0.18, Em=71, vm=0.35, Vf=0.42)
print(C)
print(S)
print(c)
print(s)

#Problem 8

def T_matrix(theta):
    """Transformation matrix to rotate stress in 2D"""
    T = np.zeros((3,3))
    T[0,0] = T[1,1] = cos(theta)**2
    T[0,1] = T[1,0] = sin(theta)**2
    T[0,2] =  2*cos(theta)*sin(theta)
    T[1,2] = -2*cos(theta)*sin(theta)
    T[2,0] = -cos(theta)*sin(theta)
    T[2,1] =  cos(theta)*sin(theta)
    T[2,2] = cos(theta)**2 - sin(theta)**2
    return T

def Tp_matrix(theta):
    """Transformation matrix to rotate strain in 2D"""
    Tp = np.zeros((3,3))
    Tp[0,0] = Tp[1,1] = cos(theta)**2
    Tp[0,1] = Tp[1,0] = sin(theta)**2
    Tp[0,2] =  cos(theta)*sin(theta)
    Tp[1,2] = -cos(theta)*sin(theta)
    Tp[2,0] = -2*cos(theta)*sin(theta)
    Tp[2,1] =  2*cos(theta)*sin(theta)
    Tp[2,2] = cos(theta)**2 - sin(theta)**2
    return Tp

def Sbar(S, theta):
    return np.linalg.inv(Tp_matrix(theta)) @ S @ T_matrix(theta)

def properties(Sb):
    Ex  = 1 / Sb[0,0]
    Gxy = 1 / Sb[2,2]
    vxy = -Ex * Sb[0,1]
    return Ex, Gxy, vxy



thetas = np.deg2rad(np.arange(0, 91, 1))

Ex, Gxy, vxy = [], [], []
for th in thetas:
    e, g, v = properties(Sbar(s, th))
    Ex.append(e); Gxy.append(g); vxy.append(v)

thetas_deg = np.arange(0, 91, 1)

plt.figure(1)
plt.plot(thetas_deg, Ex)
plt.xlabel("theta (deg)")
plt.ylabel("Ex (GPa)")
plt.grid(True)
plt.show()

plt.figure(2)
plt.plot(thetas_deg, Gxy)
plt.xlabel("theta (deg)")
plt.ylabel("Gxy (GPa)")
plt.grid(True)
plt.show()

plt.figure(3)
plt.plot(thetas_deg, vxy)
plt.xlabel("theta (deg)")
plt.ylabel("vxy")
plt.grid(True)
plt.show()



