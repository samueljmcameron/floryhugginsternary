import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

from chemeq3 import ChemEq3_2phase, ChemEq3_initialise




chi_12 = 4


# initial region of binodal
# to get this, use the symmetry of phi_1 and phi_2
# (since chi_11 = chi_22) to find the
# common tangent construction along the line
# phi_1 = phi_2.

ini = ChemEq3_initialise(chi_12 = chi_12)


x0 = [0.05,0.45]
roots = root(ini.commontangent_eqns,x0,tol=1e-10)

phi_1A,phi_1B = roots.x

print(phi_1A,phi_1B)

init_config = {'phi_1A' : phi_1A,
               'phi_2A' : phi_1A,
               'phi_1B' : phi_1B,
               'phi_2B' : phi_1B}



phi_1As = np.linspace(phi_1A,0.6,num=200,endpoint=True)

total_num = len(phi_1As)
chi_surf = np.empty([total_num],float)
phi_1A_surf = np.empty([total_num],float)
phi_2A_surf = np.empty([total_num],float)
phi_1B_surf = np.empty([total_num],float)
phi_2B_surf = np.empty([total_num],float)


count = 0

for j,phi_1A in enumerate(phi_1As):


    ch = ChemEq3_2phase(phi_1A = phi_1A,
                        chi_12 = chi_12)

    if j == 0:
        phi_2A = init_config['phi_2A']
        phi_1B = init_config['phi_1B']
        phi_2B = init_config['phi_2B']
        x0 = ch.phis_to_x(phi_1B,phi_2A,phi_2B)

    else:
        x0 = [roots[0],roots[1],roots[2]]



    solution = root(ch.rootfind_eqns,x0,
                    jac=ch.rootfind_jacobian,
                    tol=1e-8)

    roots = solution.x

    phi_1s,phi_2s = ch.get_phi1s_phi2s(roots)

    if (np.abs(phi_1s[0]-phi_1s[1])>1e-3
        and solution.success):

        phi_1B = phi_1s[1]
        phi_2A = phi_2s[0]
        phi_2B = phi_2s[1]
        chitmp = chi_12
        print(f"chi = {chi_12}, phi_1A = {phi_1A}")
        print(ch.rootfind_eqns(roots))
        print(solution.fun)
    else:
        phi_1A = np.nan
        phi_1B = np.nan
        phi_2A = np.nan
        phi_2B = np.nan
        chitmp = np.nan


    chi_surf[count] = chitmp
    phi_1A_surf[count] = phi_1A
    phi_2A_surf[count] = phi_2A
    phi_1B_surf[count] = phi_1B
    phi_2B_surf[count] = phi_2B

    count += 1

data = np.concatenate(([chi_surf],[phi_1A_surf],
                      [phi_2A_surf],[phi_1B_surf],
                      [phi_2B_surf]))

np.savetxt("data/chi_is4_phi_1A_forwards.dat",data)



# second region


phi_1As = np.linspace(0.01,init_config['phi_1A'],
                      num=200,endpoint=False)



total_num = len(phi_1As)
chi_surf = np.empty([total_num],float)
phi_1A_surf = np.empty([total_num],float)
phi_2A_surf = np.empty([total_num],float)
phi_1B_surf = np.empty([total_num],float)
phi_2B_surf = np.empty([total_num],float)


count = 0

for j,phi_1A in enumerate(phi_1As[::-1]):


    ch = ChemEq3_2phase(phi_1A = phi_1A,
                        chi_12 = chi_12)

    if j == 0:
        phi_2A = init_config['phi_2A']
        phi_1B = init_config['phi_1B']
        phi_2B = init_config['phi_2B']
        x0 = ch.phis_to_x(phi_1B,phi_2A,phi_2B)

    else:
        x0 = [roots[0],roots[1],roots[2]]



    solution = root(ch.rootfind_eqns,x0,
                    jac=ch.rootfind_jacobian,
                    tol=1e-8)

    roots = solution.x

    phi_1s,phi_2s = ch.get_phi1s_phi2s(roots)

    if (np.abs(phi_1s[0]-phi_1s[1])>1e-3
        and solution.success):

        phi_1B = phi_1s[1]
        phi_2A = phi_2s[0]
        phi_2B = phi_2s[1]
        chitmp = chi_12
        print(f"chi = {chi_12}, phi_1A = {phi_1A}")
        print(ch.rootfind_eqns(roots))
        print(solution.fun)
    else:
        phi_1A = np.nan
        phi_1B = np.nan
        phi_2A = np.nan
        phi_2B = np.nan
        chitmp = np.nan


    chi_surf[count] = chitmp
    phi_1A_surf[count] = phi_1A
    phi_2A_surf[count] = phi_2A
    phi_1B_surf[count] = phi_1B
    phi_2B_surf[count] = phi_2B

    count += 1

data = np.concatenate(([chi_surf],[phi_1A_surf],
                      [phi_2A_surf],[phi_1B_surf],
                      [phi_2B_surf]))

np.savetxt("data/chi_is4_phi_1A_backwards.dat",data)

# final region
# initial condition is tricky to get
# just trial and error

init_config = {'phi_2A' : 0.26,
               'phi_1B' : 0.23,
               'phi_2B' : 0.57}



phi_1As = np.linspace(0.037,0.09,num=200,endpoint=True)



total_num = len(phi_1As)
chi_surf = np.empty([total_num],float)
phi_1A_surf = np.empty([total_num],float)
phi_2A_surf = np.empty([total_num],float)
phi_1B_surf = np.empty([total_num],float)
phi_2B_surf = np.empty([total_num],float)


count = 0

for j,phi_1A in enumerate(phi_1As):


    ch = ChemEq3_2phase(phi_1A = phi_1A,
                        chi_12 = chi_12)

    if j == 0:
        phi_2A = init_config['phi_2A']
        phi_1B = init_config['phi_1B']
        phi_2B = init_config['phi_2B']
        x0 = ch.phis_to_x(phi_1B,phi_2A,phi_2B)

    else:
        x0 = [roots[0]+0.005,roots[1]-0.005,roots[2]]



    solution = root(ch.rootfind_eqns,x0,
                    jac=ch.rootfind_jacobian,
                    tol=1e-8)

    roots = solution.x

    phi_1s,phi_2s = ch.get_phi1s_phi2s(roots)

    if (np.abs(phi_1s[0]-phi_1s[1])>1e-3
        and solution.success):

        phi_1B = phi_1s[1]
        phi_2A = phi_2s[0]
        phi_2B = phi_2s[1]
        chitmp = chi_12
        print(f"chi = {chi_12}, phi_1A = {phi_1A}")
        print(ch.rootfind_eqns(roots))
        print(solution.fun)
    else:
        phi_1A = np.nan
        phi_1B = np.nan
        phi_2A = np.nan
        phi_2B = np.nan
        chitmp = np.nan


    chi_surf[count] = chitmp
    phi_1A_surf[count] = phi_1A
    phi_2A_surf[count] = phi_2A
    phi_1B_surf[count] = phi_1B
    phi_2B_surf[count] = phi_2B

    count += 1

data = np.concatenate(([chi_surf],[phi_1A_surf],
                      [phi_2A_surf],[phi_1B_surf],
                      [phi_2B_surf]))

np.savetxt("data/chi_is4_phi_1A_backwards_extra.dat",data)
