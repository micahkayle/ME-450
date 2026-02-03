import numpy as np
from numpy import cos,sin
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

def Sbar(S, theta):
    return np.linalg.inv(Tp_matrix(theta)) @ S @ T_matrix(theta)

def Cbar(C, theta):
    return np.linalg.inv(T_matrix(theta)) @ C @ Tp_matrix(theta)

C = np.array([
    [45.0, 4.2, 0.0],
    [4.2, 11.3, 0.0],
    [0.0, 0.0, 3.2]
])

print("C matrix at 60 degrees:")
C_60 = Cbar(C, np.pi/3)
print(C_60)

print("C matrix at 120 degrees:")
C_120 = Cbar(C, 2*np.pi/3)
print(C_120)