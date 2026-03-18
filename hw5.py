import composite_functions as cf
import numpy as np
import matplotlib.pyplot as plt

def problem1():

    #B)

    #carbon and glass fiber properties
    Ecf=350e9 #Pa
    nucf=0.2
    Egf=72e9 #Pa
    nugf=0.25
    Em=2.7e9 #Pa
    num=0.41
    Vcf=0.55
    Vgf=0.65

    C_cf, S_cf=cf.HalpinTsai_composite_properties(Ecf, nucf, Em, num, Vcf)
    C_gf, S_gf=cf.HalpinTsai_composite_properties(Egf, nugf, Em, num, Vgf)

    
    theta_list=np.array([0, 20, 90, -20, 0, 20, 90, -20])
    C_list=np.array([C_cf, C_gf, C_cf, C_gf, C_cf, C_gf, C_cf, C_gf])
    thicknesses=np.array([0.15e-3]*len(theta_list)) #m
    z_list=cf.build_z_list(thicknesses)
    
    A=cf.A_matrix(C_list, theta_list, z_list)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print("A matrix for carbon fiber composite:")
    print(A)

    B=cf.B_matrix(C_list, theta_list, z_list)
    print("B matrix for carbon fiber composite:")
    print(B)

    D=cf.D_matrix(C_list, theta_list, z_list)
    print("D matrix for carbon fiber composite:")
    print(D)

    CLT=cf.CLT(A, B, D)
    print("CLT for carbon fiber composite:")    
    print(CLT)


    #C)
    
    N=np.array([30, 42, 5])*1e3 #N/m
    M=np.array([12, 4, 2]) #N*m/m

    eps0, kappa=cf.strain_and_curvature(N, M, A, B, D)
    print("Mid-plane strains for composite:")      
    print(eps0*1e6) #convert to microstrain
    print("Curvatures for composite:")
    print(kappa)

    #D)

    z_plot = []
    sigma_x_plot = []
    sigma_y_plot = []
    tau_xy_plot = []

    for k in range(len(theta_list)):
        z_bot = z_list[k]
        z_top = z_list[k+1]

        sigma_bot = cf.Cbar(C_list[k], theta_list[k]) @ (eps0 + z_bot * kappa)
        sigma_top = cf.Cbar(C_list[k], theta_list[k]) @ (eps0 + z_top * kappa)

        z_plot.extend([z_bot, z_top])
        sigma_x_plot.extend([float(sigma_bot[0]), float(sigma_top[0])])
        sigma_y_plot.extend([float(sigma_bot[1]), float(sigma_top[1])])
        tau_xy_plot.extend([float(sigma_bot[2]), float(sigma_top[2])])




    #plot stress vs z list
    plt.plot(z_plot,np.array(sigma_x_plot)/1e6, label="Sigma_x")
    plt.plot(z_plot,np.array(sigma_y_plot)/1e6, label="Sigma_y")
    plt.plot(z_plot,np.array(tau_xy_plot)/1e6, label="Tau_xy")
    plt.xlabel("z (m)")
    plt.ylabel("Stress (MPa)")
    plt.title("Stress Distribution in Laminate")
    plt.legend()
    plt.show()




def problemtestcase():
    #test case for CLT and strain_and_curvature functions

    Ecf=350e9 #Pa
    nucf=0.2
    Egf=72e9 #Pa
    nugf=0.25
    Em=2.7e9 #Pa
    num=0.41
    Vcf=0.55
    Vgf=0.65

    C_cf, S_cf=cf.HalpinTsai_composite_properties(Ecf, nucf, Em, num, Vcf)

    theta_list=np.array([0, 45, -45, 90])
    C_list=np.array([C_cf, C_cf, C_cf, C_cf])
    thicknesses=np.array([0.5e-3, 1e-3, 1e-3, 0.5e-3]) #m
    z_list=cf.build_z_list(thicknesses) 

    A=cf.A_matrix(C_list, theta_list, z_list)
    np.set_printoptions(precision=10, suppress=True, linewidth=200)
    B=cf.B_matrix(C_list, theta_list, z_list)
    D=cf.D_matrix(C_list, theta_list, z_list)

    CLT=cf.CLT(A, B, D)
    print("CLT for carbon fiber composite:")    
    print(CLT)


    N=np.array([10, 10, 5])*1e3 #N/m
    M=np.array([15, 10, 5]) #N*m/m

    eps0, kappa=cf.strain_and_curvature(N, M, A, B, D)
    print("Mid-plane strains for test case:")      
    print(eps0)
    print("Curvatures for test case:")
    print(kappa)


#)problem controller
if __name__ == "__main__":
    problem1() 