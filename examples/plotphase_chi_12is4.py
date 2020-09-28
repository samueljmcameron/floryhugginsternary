import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root



fig = plt.figure()

ax = fig.add_subplot(111)


colors = ['r','r','r']

data1 = np.loadtxt("data/chi_is4_phi_1A_forwards.dat")
data2 = np.loadtxt("data/chi_is4_phi_1A_backwards.dat")
data3 = np.loadtxt("data/chi_is4_phi_1A_backwards_extra.dat")
for i,data in enumerate([data1,data2,data3]):


    chis = data[0]
    phi_1As = data[1]
    phi_2As = data[2]
    phi_1Bs = data[3]
    phi_2Bs = data[4]

    print(phi_1As.shape)
    print(phi_2As.shape)

    mask1 = ~np.isnan(chis)

    mask2 = np.abs(phi_1As-phi_1Bs) > 1e-3

    mask = mask1*mask2

    chis = chis[mask]
    phi_1As = phi_1As[mask]
    phi_2As = phi_2As[mask]
    phi_1Bs = phi_1Bs[mask]
    phi_2Bs = phi_2Bs[mask]


    for j in range(len(phi_1As)):
        ax.plot([phi_1As[j],phi_1Bs[j]],[phi_2As[j],phi_2Bs[j]],
                color=colors[i],marker='.')



from spinodal import Spinodal

spin = Spinodal()

phi_1s = np.linspace(0.01,0.9,num=101,endpoint=True)
phi_2s = np.linspace(0.01,0.9,num=101,endpoint=True)

XX,YY = np.meshgrid(phi_1s,phi_2s)
X,Y = np.where(XX+YY<1,XX,0),np.where(XX+YY<1,YY,0)

ZZ = spin.chi_12_spinodal(X,Y)

ax.set_xlim(0,0.6)
ax.set_ylim(0,0.6)
ax.set_xlabel(r"$\phi_1$")
ax.set_ylabel(r"$\phi_2$")
ax.set_title(r"$\chi_{12}=4$"+', '
             +r"$\chi_{11}=1$"+', '
             +r"$\chi_{22}=1$")

ax.plot(np.nan,np.nan,'.',color=colors[0],
        label='binodal')
ax.plot(np.nan,np.nan,'-',color=colors[0],
        label='tie-lines')

cs = ax.contour(XX,YY,ZZ,levels=[4],zorder=5)
cs.collections[0].set_label('spinodal')
ax.legend(frameon=False)
fig.savefig('example_chi_12_is4_chi_11ischi_22is1.png')
plt.show()
