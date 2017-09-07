# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:41:35 2016

@author: feihoo87
"""

import numpy as np
from scipy import constants as const
PHI0 = const.value('mag. flux quantum')

from FourierGrid import QuantumSystem, Noise
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

class rfSQUID(QuantumSystem):
    def __init__(self):
        super(rfSQUID, self).__init__(dim=1)
        self.set_grid(size=601, bound=[-0*np.pi, 2.4*np.pi])
        self.noise = Noise(1e-9, 20e9, amp=0.1, T=1, seed=1)

        self.energy_unit = const.h*1e9     # in GHz
        self.time_unit = 1e-9              # in ns
        L, C = 800e-12, 1500e-15
        self.Ec = const.e**2/(2*C) / self.energy_unit
        self.EL = (PHI0/(2*np.pi))**2/L / self.energy_unit
        self.beta = 2
        self.phi_e = 0.58*2*np.pi

    def T(self, n):
        return 4*self.Ec*n**2

    def U(self, phi, time=0):
        def step_t(w1, w2, t0, width, t):
            return w1 + (w2 - w1) / (1 + np.exp(-(t-t0)/width))
        phi_e = self.phi_e + step_t(0, 0.04*2*np.pi, 0.2, 0.01, time) - step_t(0, 0.12*2*np.pi, 0.7, 0.01, time) + self.noise.value(time*self.time_unit)
        return self.EL*(0.5*(phi - phi_e)**2-self.beta*np.cos(phi))

def testrfSQUID():
    q = rfSQUID()
    E, s = q.States()
    x, = q.grid.x()
    x = np.linspace(x[0], x[-1], len(s[:,0]))
    plt.plot(x, q.U(x))
    print('w_01 = %g GHz' % (E[1]-E[0]))
    for i in range(100):
        psi = s[:,i]
        p = (psi*psi.conj()).cumsum()
        mark = (p>0.001)*(p<0.999)
        #plt.plot(x[mark], 10*np.imag(psi[mark])+E[i], color='blue', alpha=0.7, lw=1)
        plt.hlines(E[i], x[mark].min(), x[mark].max(), color='black')
        plt.plot(x[mark], 10*np.real(psi[mark])+E[i], color='red', alpha=0.7, lw=1)
        plt.fill_between(x[mark], E[i]+10*np.real(psi[mark]), E[i]-0*np.abs(psi[mark]), color='red', alpha=0.2)
    plt.show()

anim = None
def testDynamic():
    q = rfSQUID()
    x, = q.grid.x()
    tlist = np.linspace(0,1,10001)
    psi0 = np.exp(-0.5*((x-1.66)/0.2)**2)
    #npz = np.load('dynamic3.npz')
    #psi0 = npz['ret'][-1,:]
    #print((psi0.conj()*psi0).sum())
    psi0 /= (psi0.conj()*psi0).sum()
    ret, U_list = q.Dynamic(psi0, tlist, time_dependent=True, with_U=True)
    np.savez_compressed('dynamic.npz', tlist=tlist, psi0=psi0, x=x, ret=ret, U=U_list)
    #np.savez_compressed('dynamic4.npz', tlist=tlist+npz['tlist'][-1], psi0=psi0, x=x, ret=ret)

class nSQUID(QuantumSystem):
    def __init__(self, C=1100e-15, beta=5.4, M=383e-12, L=469e-12, L0=45e-12,
                 xlim = [-0.9, 0.9], ylim = [-0.3, 0.3], grid_size = [101, 31], bias=[0.5, 0]):
        super(nSQUID, self).__init__(dim=2)
        self.set_grid(size=grid_size,
            bound=[list(map(lambda x:2*np.pi*x, xlim)),
                   list(map(lambda x:2*np.pi*x, ylim))])

        self.energy_unit = const.h*1e9      # in GHz
        self.beta = beta
        self.rate = (L+M)/(L-M+2*L0)
        self.EL = 2*(PHI0/2/np.pi)**2/(L+M) / self.energy_unit
        self.Ec = 4*const.e**2/(2*C) / self.energy_unit

        self.set_bias(*bias)

    def set_bias(self, theta_e, phi_e):
        self.args = [theta_e*2*np.pi, phi_e*2*np.pi]

    def U(self, theta, phi):
        theta_e, phi_e = self.args
        return (-self.beta*np.cos(phi+phi_e)*np.cos(theta+theta_e)
               +0.5*theta**2
               +0.5*self.rate*phi**2)*self.EL

    def T(self, k1, k2):
        return 0.5*(k1**2+k2**2)*self.Ec

    def plot_U(self, clip=True):
        grid_x, grid_y = self.grid.x(sparse=False)
        if clip:
            sp, pos = self.saddle_points(self.grid.U(self.U, clip=False))
            ep, pos = self.extreme_points(self.grid.U(self.U, clip=False))
            U = self.grid.U(self.U, clip=True)
            vmin, vmax = U.min(), U.max()
            if len(sp)>0:
                print(sp, pos)
                vmax = max(vmax, *sp)
            if len(ep)>0:
                vmin = ep[0] if len(ep)==1 else min(*ep)
            U = self.grid.U(self.U, vmin=vmin, vmax=vmax, clip=True)
        else:
            U = self.grid.U(self.U, clip=False)
        fig = plt.figure()
        ax = Axes3D(fig)
        X, Y, Z = grid_x/2/np.pi, grid_y/2/np.pi, U
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
        #cset = ax.contour(X, Y, Z)
        #x, y, z = grid_x.flatten()/2/np.pi, grid_y.flatten()/2/np.pi, U.flatten()
        #ax.plot_trisurf(x[z<U.max()],y[z<U.max()],z[z<U.max()], cmap=plt.cm.hot)
        #ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
        ax.set_xlabel('theta')
        ax.set_ylabel('phi')
        #ax.set_zlim([U.min(), 382])
        # savefig('../figures/plot3d_ex.png',dpi=48)
        plt.show()

    def plot_states(self, E, s, cols=3, rows=3):
        pos_lst = [[i,j] for i in range(rows) for j in range(cols)]
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, squeeze=False)
        extent = list(np.array(self.grid.bound).flatten()/2/np.pi)
        s_lim = np.abs(s[:,cols*rows]).max()
        for i, pos in enumerate(pos_lst):
            ax = axes[pos[0],pos[1]]
            [s.set_visible(False) for loc,s in ax.spines.items()]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(np.real(s[:,i]).reshape(*self.grid.size).T,
                cmap=plt.get_cmap('RdBu'), vmin=-s_lim, vmax=s_lim,
                origin='lower', extent=extent, aspect='equal')
            ax.set_xlabel('$E_{%d}=%g \mathrm{GHz}$'%(i, E[i]))
        plt.show()

    def saddle_points(self, U):
        X_size, Y_size = self.grid.size
        ret = []
        pos = []
        for i in range(Y_size-2):
            for j in range(X_size-2):
                if ((U[i,j+1]+U[i+2,j+1]-2*U[i+1,j+1])*(U[i+1,j]+U[i+1,j+2]-2*U[i+1,j+1]) < 0 \
                and (U[i+1,j+1]-U[i,j+1])*(U[i+1,j+1]-U[i+2,j+1]) > 0 and (U[i+1,j+1]-U[i+1,j])*(U[i+1,j+1]-U[i+1,j+2]) > 0) \
                or ((U[i,j]+U[i+2,j+2]-2*U[i+1,j+1])*(U[i,j+2]+U[i+2,j]-2*U[i+1,j+1]) < 0 \
                and (U[i+1,j+1]-U[i,j])*(U[i+1,j+1]-U[i+2,j+2]) > 0 and (U[i+1,j+1]-U[i+2,j])*(U[i+1,j+1]-U[i,j+2]) > 0):
                    ret.append(U[i+1,j+1])
                    pos.append([i+1, j+1])
        return ret, pos

    def extreme_points(self, U):
        X_size, Y_size = self.grid.size
        ret = []
        pos = []
        for i in range(Y_size-2):
            for j in range(X_size-2):
                if ((U[i,j+1]+U[i+2,j+1]-2*U[i+1,j+1])*(U[i+1,j]+U[i+1,j+2]-2*U[i+1,j+1]) > 0 \
                and (U[i+1,j+1]-U[i,j+1])*(U[i+1,j+1]-U[i+2,j+1]) > 0 and (U[i+1,j+1]-U[i+1,j])*(U[i+1,j+1]-U[i+1,j+2]) > 0) \
                or ((U[i,j]+U[i+2,j+2]-2*U[i+1,j+1])*(U[i,j+2]+U[i+2,j]-2*U[i+1,j+1]) > 0 \
                and (U[i+1,j+1]-U[i,j])*(U[i+1,j+1]-U[i+2,j+2]) > 0 and (U[i+1,j+1]-U[i+2,j])*(U[i+1,j+1]-U[i,j+2]) > 0):
                    ret.append(U[i+1,j+1])
                    pos.append([i+1, j+1])
        return ret, pos

def testnSQUID():
    q = nSQUID(C=1100e-15, beta=5.4, M=383e-12, L=469e-12, L0=45e-12,
               xlim = [-0.18, -0.06], ylim = [0.12, 0.18],
               grid_size = [101, 51], bias = [0.7, 0.15])
    #q.set_bias(0.7, 0.15)
    q.plot_U()
    #E, s = q.States()
    #q.plot_states(E, s, cols=6, rows=6)

class Tranamon(QuantumSystem):
    def __init__(self):
        super(Tranamon, self).__init__(dim=1)
        self.set_grid(size=1001, bound=[-500, 500])
        C  = 1000e-15
        Ic = 1e-6
        self.energy_unit = const.h*1e9     # GHz
        self.EJ = Ic*PHI0/(2*np.pi) / self.energy_unit
        self.Ec = const.e**2/C / self.energy_unit
        self.ng = 0.5

    def T(self, phi):
        return -self.EJ * np.cos(phi)

    def U(self, nn):
        return 4*self.Ec * (-nn-self.ng)**2

def testTransmon():
    q = Tranamon()
    x = np.linspace(-1,1,101)
    N = 5
    levels = []
    for ng in x:
        q.ng = ng
        E,s = q.States()
        levels.append(E[:N])
        print(ng)
    levels = np.array(levels)
    for i in range(N):
        plt.plot(x, levels[:,i])
    plt.show()

def spec(comm=0.16, xlim=[-0.2, 0.0], ylim=[0.1, 0.2], grid_size=[101, 51]):
    L, M, L0 = 469e-12, 383e-12, 45e-12
    beta = 5.4
    C = 1100e-15
    q = nSQUID(C=C, beta=beta, M=M, L=L, L0=L0,
               xlim = xlim, ylim = ylim,
               grid_size = grid_size)

    x = np.linspace(0.6, 0.72, 13)
    res = []
    fname = 'spec_diff_%.2g_%.2g_comm_%.2g' % (x.min(), x.max(), comm)
    for diff in x:
        q.set_bias(diff, comm)
        E,s = q.States()
        states_fname = '%s/diff=%g_comm=%g.npz' % (fname, diff, comm)
        np.savez_compressed(states_fname, E=E, s=s,
                            xlim=np.array([xlim]),
                            ylim=np.array([ylim]),
                            grid_size=np.array([grid_size]),
                            C=C, L=L, M=M, L0=L0, beta=beta)
        res.append(E)
        print('diff = %g: w_01 = %g GHz, w_12 = %g GHz' % (diff, E[1]-E[0], E[2]-E[1]))
        np.savetxt(('%s.txt'%fname), np.array(res))
    res = np.array(res)

    for i in range(100):
        plt.plot(x, res[:,i], color='blue', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    #qubit = nSQUID(C=1100e-15, beta=5.1, M=383e-12, L=469e-12, L0=45e-12,
    #               xlim = [-0.7, -0.38], ylim = [-0.2, 0.1], grid_size = [51, 47],
    #               bias = [0.67, 0.13])
    #qubit.plot_U()
    testnSQUID()
    #spec()
    #testTransmon()
    #testrfSQUID()
    #testDynamic()
