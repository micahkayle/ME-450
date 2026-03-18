import numpy as np
from numpy import cos,sin
import matplotlib.pyplot as plt

def ROM_composite_properties(Ef, nuf, Em, num, Vf):

    Vm = 1 - Vf
    
    Gf = Ef / (2*(1+nuf))
    Gm = Em / (2*(1+num))
    
    E1 = Ef*Vf + Em*Vm
    E2 = 1 / (Vf/Ef + Vm/Em) #ROM
    
    G12 = 1 / (Vf/Gf + Vm/Gm) #ROM

    G12=1/(Vf/Gf+(1-Vf)/Gm)
    
    nu12 = nuf*Vf + num*Vm
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

def HalpinTsai_composite_properties(Ef, nuf, Em, num, Vf):

    Vm = 1 - Vf
    
    Gf = Ef / (2*(1+nuf))
    Gm = Em / (2*(1+num))
    
    E1 = Ef*Vf + Em*Vm
    xi_E = 2.0
    eta_E = (Ef/Em - 1) / (Ef/Em + xi_E)
    E2 = Em * (1 + xi_E*eta_E*Vf) / (1 - eta_E*Vf) #Halpin-Tsai
    
    xi_G = 1.0
    eta_G = (Gf/Gm - 1) / (Gf/Gm + xi_G)
    G12 = Gm * (1 + xi_G*eta_G*Vf) / (1 - eta_G*Vf) #Halpin-Tsai
    
    nu12 = nuf*Vf + num*Vm
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



def tsai_wu_criterion(sigma1t,sigma1c,sigma2t,sigma2c,tau12t,sigma_bar):
    sigma1,sigma2,tau12=sigma_bar
    F12 = 0 #Assume no interaction between sigma1 and sigma2
    I=((sigma1**2)/(sigma1t*sigma1c)+(sigma2**2)/(sigma2t*sigma2c)-
    (F12*sigma1*sigma2)/np.sqrt(sigma1t*sigma1c*sigma2t*sigma2c)+
    (tau12/tau12t)**2+sigma1*(1/sigma1t-1/sigma1c)+sigma2*(1/sigma2t-1/sigma2c)
    )
    return I 

def max_stress_criterion(sigma1t,sigma1c,sigma2t,sigma2c,tau12t,sigma_bar):
    sigma1,sigma2,tau12=sigma_bar
    return (sigma1>=sigma1t) or (sigma1<=-sigma1c) or (sigma2>=sigma2t) or (sigma2<=-sigma2c) or (abs(tau12)>=tau12t)


def build_z_list(thicknesses):
    n = len(thicknesses)
    z_list = np.zeros(n+1)
    z_list[0] = -0.5 * np.sum(thicknesses)
    for k in range(n):
        z_list[k+1] = z_list[k] + thicknesses[k]
    
    return z_list

def A_matrix(C_list, theta_list, z_list):
    #A matrix for composites
    n=len(C_list)
    A = np.zeros((3,3))
    for k in range(n):
        A += Cbar(C_list[k], theta_list[k]) * (z_list[k+1] - z_list[k])

    return A

def B_matrix(C_list, theta_list, z_list):
    #B matrix for composites
    n=len(C_list)
    B = np.zeros((3,3))
    for k in range(n):
        B += Cbar(C_list[k], theta_list[k]) * (z_list[k+1]**2 - z_list[k]**2) / 2

    return B
def D_matrix(C_list, theta_list, z_list):
    #D matrix for composites
    n=len(C_list)
    D = np.zeros((3,3))
    for k in range(n):
        D += Cbar(C_list[k], theta_list[k]) * (z_list[k+1]**3 - z_list[k]**3) / 3

    return D


def CLT(A, B, D):
    #Classical Laminate Theory
    ABD = np.block([[A, B], [B, D]])
    return ABD

def strain_and_curvature(N, M, A, B, D):
    ABD = np.block([[A, B], [B, D]])
    NM = np.concatenate((N, M))
    eps_kappa = np.linalg.solve(ABD, NM)
    eps0 = eps_kappa[:3]
    kappa = eps_kappa[3:]
    return eps0, kappa


