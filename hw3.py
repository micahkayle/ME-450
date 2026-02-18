import numpy as np
import matplotlib.pyplot as plt
import composite_functions as cf

def problem3():
    Gf=125e3
    Gm=1.6e3
    tau=60
    Vf=np.linspace(0.2,0.9,100)
    G12=1/(Vf/Gf+(1-Vf)/Gm)

    for d in [1,2,3]:
        p=np.deg2rad(d)
        plt.plot(Vf,tau/(np.tan(p)+tau/G12),label=f"{d}°")

    plt.xlabel("Vf")
    plt.ylabel("sigma_c (MPa)")
    plt.legend()
    plt.grid()
    plt.show()


def problem4():
    Vf=0.58
    Vm=1-Vf
    Em=2.8e3
    Ef=71e3
    m=13.2
    sigma0=760
    L=300
    L0=50

    #Use Rom for Ec
    Ec=Vf*Ef+Vm*Em

    #define Pf and sigmac
    def Pf(sigmac):
        sigmaf=(Ef/Ec)*sigmac
        return 1-np.exp(-(L/L0)*(sigmaf/sigma0)**m)

    def sigmac_at(P):
        sigmaf=sigma0*((-np.log(1-P))/(L/L0))**(1/m)
        return sigmaf*(Ec/Ef)

    sigmac=np.linspace(0,1000,100)
    Pf_values=Pf(sigmac)

    #plot Pf vs sigmac
    plt.plot(sigmac,100*Pf_values)
    plt.xlabel("sigma_c(MPa)")
    plt.ylabel("Percent fiber failure(%)")
    plt.grid(True)
    plt.show()

    #find sigmac at 1% and 50% fiber failure
    print("sigm_c@1%=",f"{sigmac_at(0.01):.2f}","MPa")
    print("sigma_c@50%=",f"{sigmac_at(0.50):.2f}","MPa")

    #b)
    sigma_m=45 #MPa
    epsilon_m=sigma_m/Em
    print("epsilon_m=",f"{epsilon_m:.3f}")
    sigmaf=Ef*epsilon_m
    print("sigma_f=",f"{sigmaf:.2f}","MPa")
    print("Pf at sigma_f=",f"{Pf(sigmaf):.6f}")

    #check
    epsilon_c_at_1pf=sigmac_at(0.01)/Ec
    print("epsilon_c at 1% Pf=",f"{epsilon_c_at_1pf:.3f}")


def problem5():
    sigma_bar_30=np.array([89,0,0]) #Mpa
    sigma_30=cf.T_matrix(np.deg2rad(30)) @ sigma_bar_30
    sigma_bar_45=np.array([54,0,0]) #Mpa
    sigma_45=cf.T_matrix(np.deg2rad(45)) @ sigma_bar_45
    sigma_bar_60=np.array([39,0,0]) #Mpa
    sigma_60=cf.T_matrix(np.deg2rad(60)) @ sigma_bar_60

    
    system_eq=np.array([
        [sigma_30[0]**2-sigma_30[0]*sigma_30[1], sigma_30[1]**2, sigma_30[2]**2],
        [sigma_45[0]**2-sigma_45[0]*sigma_45[1], sigma_45[1]**2, sigma_45[2]**2],
        [sigma_60[0]**2-sigma_60[0]*sigma_60[1], sigma_60[1]**2, sigma_60[2]**2]
    ])

    solved_system_eq=np.linalg.solve(system_eq, np.ones(3))
    sigma1_dot=np.sqrt(1/solved_system_eq[0])
    sigma2_dot=np.sqrt(1/solved_system_eq[1])
    tau12_dot=np.sqrt(1/solved_system_eq[2])

    print("sigma1_dot=",f"{sigma1_dot:.2f}","MPa")
    print("sigma2_dot=",f"{sigma2_dot:.2f}","MPa")
    print("tau12_dot=",f"{tau12_dot:.2f}","MPa")



def problem6():
    sigma_tests=[
        [2000,90,100],
        [1000,90,160],
        [1800,60,150],
        [-1200,60,150],
        [1200,-180,100]
    ]

    for i,s in enumerate(sigma_tests,1):
        FI=cf.tsai_wu_criterion(2100,1300,100,200,180,np.array(s))
        print(f"Case {i}: FI={FI:.3f}, Fail? {FI>=1}")

    for i,s in enumerate(sigma_tests,1):
        fail=cf.max_stress_criterion(2100,100,180,np.array(s))
        print(f"Case{i}:Fail? {fail}")

def problem7():

    sigma_bar=np.array([320,90,0]) #MPa
    theta_deg=np.arange(0,90.5,0.5)

    for d in theta_deg:
        sigma12=cf.T_matrix(np.deg2rad(d))@sigma_bar
        I=cf.tsai_wu_criterion(900,500,150,300,200,sigma12)
        if I>=1:
            print(f"Tsai-Wu failure angle≈{d:.1f}deg")
            break

    for d in theta_deg:
        sigma12=cf.T_matrix(np.deg2rad(d))@sigma_bar
        if cf.max_stress_criterion(900,500,150,300,200,sigma12):
            print(f"Max-Stress failure angle≈{d:.1f}deg")
            break

#problem controller
if __name__ == "__main__":
    problem7() 