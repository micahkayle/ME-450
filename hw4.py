
import composite_functions as cf
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)



def problem4():

    #graph of fiber pullout energy vs pullout aspect ratio
    s=np.linspace(1,60,100) #pullout aspect ratio
    r=0.5*18*1e-6 #m
    taui= 50*1e6 #Pa
    Vf=0.56
    Gcp=1/3*Vf*s**2 *r*taui

    plt.plot(s,Gcp/1000) #convert to kJ/m^2
    plt.xlabel("s")
    plt.ylabel("Gcp (kJ/m^2)")  
    plt.title("Fiber Pullout Energy vs Pullout Aspect Ratio")
    plt.grid()
    plt.show()

    
    #pullout length
    Gc=100000 #J/m^2
    s_critical=np.sqrt(3*Gc/(Vf*r*taui))
    pullout_length=r*s_critical
    print(f"Critical pullout length: {pullout_length*1e3:.2f} mm")

    #facture toughness of composite KIc
    Ef=72e9 #Pa
    Em=3.1e9 #Pa
    E1=Vf*Ef+(1-Vf)*Em

    KIc=np.sqrt(Gc*E1)
    print(f"Fracture toughness of composite KIc: {KIc/1e6:.2f} MPa*m^0.5")


def problem5a():

    #B matrix for [0(2),90(2)]
    C,S=cf.ROM_composite_properties(Ef=310e9, nuf=0.18, Em=3.2e9, num=0.39, Vf=0.62)  
    C_bar_0= cf.Cbar(C, 0)
    C_bar_90= cf.Cbar(C, 90)
    
    C_bar_stackup= (C_bar_0, C_bar_0, C_bar_90, C_bar_90)
    t_ply=1e-3 #m
    
    def B_matrix(C_bar_stackup, t_ply):
        B=np.zeros((3,3))
        n=len(C_bar_stackup)
        t=np.full(n, t_ply)
        h=n*t_ply
        z=np.linspace(-h/2, h/2, n+1)

        for k in range(len(C_bar_stackup)):
            B += C_bar_stackup[k]*(z[k+1]**2-z[k]**2)/2
        return B

    B=B_matrix(C_bar_stackup, t_ply)
    print("B matrix for [0(2),90(2)]:")
    print(B)

    kappa=np.array([0.01,-0.01,0]) #bending curvature
    N=B@kappa
    print("Nx for [0(2),90(2)]:")
    print(N[0])

def problem5b():

    #B matrix for [0,90]s = [0,90,90,0]
    C,S=cf.ROM_composite_properties(Ef=310e9, nuf=0.18, Em=3.2e9, num=0.39, Vf=0.62)  
    C_bar_0= cf.Cbar(C, 0)
    C_bar_90= cf.Cbar(C, 90)
    
    C_bar_stackup= (C_bar_0, C_bar_90, C_bar_90, C_bar_0)
    t_ply=1e-3 #m
    
    def B_matrix(C_bar_stackup, t_ply):
        B=np.zeros((3,3))
        N=len(C_bar_stackup)
        t=np.full(N, t_ply)
        h=N*t_ply
        z=np.linspace(-h/2, h/2, N+1)

        for k in range(len(C_bar_stackup)):
            B += C_bar_stackup[k]*(z[k+1]**2-z[k]**2)/2
        return B

    B=B_matrix(C_bar_stackup, t_ply)
    print("B matrix for [0,90]s = [0,90,90,0]:")
    B=cf.clean(B,tol=1e-12)
    print(B)

    kappa=np.array([0.01,-0.01,0]) #bending curvature
    N=B@kappa
    print("Nx for [0,90]s = [0,90,90,0]:")
    print(N[0])


def problem6():

    """Write a function to predict the first ply failure in a laminate. The inputs should be the global applied
    stress 𝜎𝑔, the ply layup angles, thicknesses and stiffness tensor 𝐶 and the composite strengths (𝜎1∗, 𝜎2∗
    and 𝜏12∗). It should use a Tsai-Hill criterion to predict failure. The function should output which if any
    ply has failed and the corresponding stresses in that ply."""

    """
    To check your code, take a [0º/±45º/90º]s laminate with ply thicknesses of 0.5mm for the 0º and
    90º plies and 1.0mm for the ±45º plies. The first ply to fail will be the -45°, which will fail at 𝜎𝑜 =
    82.4 𝑀𝑃𝑎 with a ply stress of 𝜎−45° = [148.71, 59.81, 10.11]𝑀𝑃𝑎.    
    """

    #a) 
    def tsai_hill_criterion(sigma_g,layup_angles,thicknesses,C,stress_failure):
        
        C_bar_stackup = [cf.Cbar(C, (theta)) for theta in layup_angles]
        
        def A_matrix(C_bar_stackup, thicknesses):
            A = np.zeros((3,3))
            h = np.sum(thicknesses)
            n = len(thicknesses)

            z = np.zeros(n+1)
            z[0] = -0.5 * np.sum(thicknesses)
            for k in range(n):
                z[k+1] = z[k] + thicknesses[k]

            for k in range(n):
                A += C_bar_stackup[k] * (z[k+1] - z[k])
            return A
    
        A=A_matrix(C_bar_stackup, thicknesses)

        h = np.sum(thicknesses)
        N = h * sigma_g
        eps0 = np.linalg.solve(A, N)
        
        FI_max = 0
        ply_max = 0
        sigma_at_max = np.zeros(3)

        for i, (theta, t) in enumerate(zip(layup_angles,thicknesses)):
            
           
            eps_local = cf.Tp_matrix(np.deg2rad(theta)) @ eps0
            sigma_local = C @ eps_local

            X, Y, S = stress_failure  # [sigma1*, sigma2*, tau12*]
            s1, s2, t12 = sigma_local
            FI = (s1/X)**2 - (s1*s2)/(X**2) + (s2/Y)**2 + (t12/S)**2

            if FI > FI_max:
                FI_max = FI
                ply_max = i
                sigma_at_max = sigma_local

        return FI_max, ply_max, sigma_at_max

    #b)
    
    #Test Case
    layup_angles_test  =[0,45,-45,90,90,-45,45,0] #deg
    thicknesses_test=[0.5e-3,1e-3,1e-3,0.5e-3,0.5e-3,1e-3,1e-3,0.5e-3] #m
    
    #Real Case
    layup_angles  =[0,30,45,60,90,90,60,45,30,0] #deg
    thicknesses=[1e-3,0.75e-3,1e-3,0.75e-3,1e-3,1e-3,0.75e-3,1e-3,0.75e-3,1e-3] #m
    
    C,S=cf.HalpinTsai_composite_properties(Ef=310e9, nuf=0.19, Em=3.2e9, num=0.39, Vf=0.65)
    stress_failure=np.array([1.9e9,60e6,140e6]) #Pa

    for sigma_o in np.linspace(0.1e6, 1000e6, 100000):
        sigma_g = np.array([3, 2, 1]) * sigma_o  #Pa
        FI_max_test, ply_max_test, sigma_at_max_test = tsai_hill_criterion(sigma_g, layup_angles_test, thicknesses_test, C, stress_failure)
        if FI_max_test >= 1.0:
            print(f"sigma_o = {sigma_o/1e6:.3f} MPa")
            print(f"failed ply ={layup_angles_test[ply_max_test]} deg)")
            print(f"ply stress [s1,s2,t12] = {sigma_at_max_test/1e6} MPa")
            break
        
   
    for sigma_o in np.linspace(0.1e6, 1000e6, 100000):
        sigma_g = np.array([3, 2, 1]) * sigma_o  #Pa     
        FI_max, ply_max, sigma_at_max = tsai_hill_criterion(sigma_g, layup_angles, thicknesses, C, stress_failure)
        if FI_max >= 1.0:
            print(f"sigma_o = {sigma_o/1e6:.3f} MPa")
            print(f"failed ply ={layup_angles_test[ply_max]} deg)")
            print("ply stress [s1,s2,t12] = ", np.round(sigma_at_max/1e6,3), "MPa")
            break
    

    
#)problem controller
if __name__ == "__main__":
    problem6() 