import numpy as np
from numpy import cos,sin
import matplotlib.pyplot as plt
#problem 4


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
def deg(theta_deg):
    return np.deg2rad(theta_deg)
def Sbar(S, theta_deg):
    return np.linalg.inv(Tp_matrix(deg(theta_deg))) @ S @ T_matrix(deg(theta_deg))

def Cbar(C, theta_deg):
    return np.linalg.inv(T_matrix(deg(theta_deg))) @ C @ Tp_matrix(deg(theta_deg))

def clean(M, tol=1e-12):
    M = M.copy()
    M[np.abs(M) < tol] = 0.0
    return M



C4 = np.array([
    [45.0, 4.2, 0.0],
    [4.2, 11.3, 0.0],
    [0.0, 0.0, 3.2]
])

C_bar_lam=1/3*(Cbar(C4, 0) + Cbar(C4, 60) + Cbar(C4, 120))
S_bar_lam=np.linalg.inv(C_bar_lam)

C_bar_lam = clean(C_bar_lam)
S_bar_lam = clean(S_bar_lam)

stress_g=np.array([0.030, 0.020, 0.010])  #GPa

strain_g = S_bar_lam @ stress_g

stress_60_g=Cbar(C4, 60) @ strain_g

stress_60_m=T_matrix(np.pi/3) @ stress_60_g

#print("strain_g:", strain_g)

#print("stress_60_g:", stress_60_g)

#print("stress_60_m:", stress_60_m)

print("C_bar_lam:\n", C_bar_lam)
print("S_bar_lam:\n", S_bar_lam)

#Problem 5

C5 = np.array([
    [210.0, 6.1, 0.0],
    [6.1, 21.5, 0.0],
    [0.0, 0.0, 5.6]
])

C_bar_lam5_1=1/4*(Cbar(C5, 0) + Cbar(C5, 45) + Cbar(C5, 90) + Cbar(C5, 135))

C_bar_lam5_2=1/2*(Cbar(C5, 0) + Cbar(C5, 90))

print("c_bar_lam5_1:\n", clean(C_bar_lam5_1))
print("c_bar_lam5_2:\n", clean(C_bar_lam5_2))



#Problem 6

Ef = 280  #GPa
Em = 3.4   #GPa  
Vf = 0.38
s = np.linspace(1, 40, 100)
n = 0.1

E1 =Vf*Ef*(1-((Ef-Em)*np.tanh(n*s))/(Ef*n*s))+(1-Vf)*Em

E_rom =Vf*Ef+(1-Vf)*Em


plt.figure()
plt.plot(s, E1, label="Modified shear lag")
plt.plot(s, np.ones_like(s)*E_rom, label="Rule of mixtures")
plt.xlabel("Aspect ratio s")
plt.ylabel("E1(GPa)")
plt.legend()
plt.show()


#Problem 7
#a)
def C_bar_lam(C, theta_list, t_list):
    h = sum(t_list)
    return sum(t * Cbar(C, theta) for theta, t in zip(theta_list, t_list)) / h

#b)
C7 = np.array([
    [198.1, 4.25, 0.0],
    [4.25, 13.9, 0.0],
    [0.0, 0.0, 3.7]
])

theta_list = np.array([0, 90, 60, -60, 30,-30]) #degrees
t_list = np.array([1, 1, 0.5, 0.5, 0.5, 0.5])  #mm

print("C_bar_lam for the laminate:\n", clean(C_bar_lam(C7, theta_list, t_list)))

#test case

theta_list_test = np.array([0, 45, 90, -45]) #degrees
t_list_test = np.array([0.5, 0.5, 0.5, 0.5])  #mm

print("C_bar_lam for the test case:\n", clean(C_bar_lam(C7, theta_list_test, t_list_test)))

#test case 2
t_list_test2 = np.array([0.5, 1, 0.5, 1])  #mm

print("C_bar_lam for the test case 2:\n", clean(C_bar_lam(C7, theta_list_test, t_list_test2)))


#Problem 8
strain_g=np.array([150e-6,150e-6,100e-6])
strain_90_12=Tp_matrix(np.pi/2) @ strain_g
stress_90_12=Cbar(C7, 90) @ strain_90_12

angles=np.arange(0,91,10)

def rotated_stress_angles(stress_local, angles):
    return np.array([T_matrix(deg(a)) @ stress_local for a in angles])

stresses_angles = rotated_stress_angles(stress_90_12, angles)

stress_xk = stresses_angles[:, 0]

plt.plot(angles, stress_xk * 1000, marker='o')
plt.xlabel("theta (deg)")
plt.ylabel("sigma_xk (MPa)")
plt.show()