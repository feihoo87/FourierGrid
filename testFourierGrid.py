# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:41:35 2016

@author: feihoo87
"""
__all__ = ['FourierGrid', 'QuantumSystem', 'Noise']

import numpy as np
from scipy import constants as const
PHI0 = const.value('mag. flux quantum')
from FourierGrid import QuantumSystem, wigner
import matplotlib.pyplot as plt

class Cavity(QuantumSystem):
    def __init__(self, C=1000e-15, L=1000e-12):
        super(Cavity, self).__init__(dim=1)
        self.C = C
        self.L = L
        self.energy_unit = const.h*1e9   # in GHz
        print("f_10 = %g GHz" % (1.0e-9/np.sqrt(L*C)/2/np.pi))
        self.set_grid(size=501, bound=[-1, 1])

    def T(self, k):
        return 4*const.e**2/(2*self.C)/self.energy_unit *k**2

    def U(self, x):
        return 0.5*x**2/self.L*(PHI0/2/np.pi)**2/self.energy_unit

def testCavity():
    q = Cavity()
    E, s = q.States()
    x, = q.grid.x()
    x = np.linspace(x[0], x[-1], len(s[:,0]))
    plt.plot(x, q.U(x))
    print('w_01 = %g GHz' % (E[1]-E[0]))
    for i in range(10):
        plt.hlines(E[i], x[0], x[-1], color='black')
        plt.plot(x, 30*np.real(s[:,i])+E[i], color='red', alpha=0.7)
    plt.show()

    '''
    cols, rows = 1, 1

    pos_lst = [[i,j] for i in range(rows) for j in range(cols)]
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, squeeze=False)
    #s_lim = np.abs(s[:,cols*rows]).max()
    for i, pos in enumerate(pos_lst):
        ax = axes[pos[0],pos[1]]
        [s.set_visible(False) for loc,s in ax.spines.items()]
        ax.set_xticks([])
        ax.set_yticks([])
        wig = wigner(s[:,i])
        s_lim = np.max(np.abs(wig))
        ax.imshow(wig, vmin=-s_lim, vmax=s_lim,
            cmap=plt.get_cmap('RdBu'),
            origin='lower', aspect='equal')
        ax.set_xlabel('$E_{%d}=%g$'%(i, E[i]))
    plt.show()
    '''

class HarmonicOscillator2D(QuantumSystem):
    def __init__(self):
        super(HarmonicOscillator2D, self).__init__(dim=2)
        self.set_grid(size=[51,51], bound=[[-6,6],[-6,6]])

    def T(self, k1, k2):
        return 0.5*(k1**2+k2**2)

    def U(self, x, y):
        rr = x**2+y**2
        ret = 0.5*rr
        #ret[rr>(5.9**2)] = 1e15
        return ret

    def plot_states(self, E, s, cols=3, rows=3):
        pos_lst = [[i,j] for i in range(rows) for j in range(cols)]
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, squeeze=False)
        extent = list(np.array(self.grid.bound).flatten())
        s_lim = np.abs(s[:,cols*rows]).max()
        for i, pos in enumerate(pos_lst):
            ax = axes[pos[0],pos[1]]
            [s.set_visible(False) for loc,s in ax.spines.items()]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(np.real(s[:,i]).reshape(*self.grid.size).T,
                cmap=plt.get_cmap('RdBu'), vmin=-s_lim, vmax=s_lim,
                origin='lower', extent=extent, aspect='equal')
            ax.set_xlabel('$E_{%d}=%g$'%(i, E[i]))
        plt.show()

def testHarmonicOscillator2D():
    q = HarmonicOscillator2D()
    E, s = q.States()
    q.plot_states(E, s, cols=6, rows=6)

class Hydrogen(QuantumSystem):
    def __init__(self):
        super(Hydrogen, self).__init__(dim=3)
        self.set_grid(size=[16,16,16], bound=[[-4,4],[-4,4],[-4,4]])
        self.energy_unit = const.e                                          # eV
        mu = 1/(1/const.m_e + 1/const.m_p)
        self.a_mu = 4*np.pi*const.epsilon_0*const.hbar**2/(mu*const.e**2)   # length unit
        self.Ek = const.hbar**2/(2*mu*self.a_mu**2) / self.energy_unit
        self.Ev = const.e**2/(4*np.pi*const.epsilon_0*self.a_mu) / self.energy_unit
        print('''
        mu = %g kg
        Ek = %g eV
        Ev = %g eV
        a_mu = %g m
        ''' % (mu, self.Ek, self.Ev, self.a_mu))

    def T(self, k1, k2, k3):
        return (k1**2+k2**2+k3**2)*self.Ek

    def U(self, x1, x2, x3):
        r = np.sqrt(x1**2+x2**2+x3**2)
        return -self.Ev/r

def testHydrogen():
    h = Hydrogen()
    E, s = h.States()
    fig, axes = plt.subplots(1,3, sharex=True, figsize=(2*3,2*1))
    for i in range(3):
        psi = np.real(s[:,i]).reshape(*h.grid.size)
        axes[i].imshow(psi.sum(axis=1))
    plt.show()

class ParticleInABox(QuantumSystem):
    def __init__(self):
        super(ParticleInABox, self).__init__(dim=1)
        self.set_grid(size=1001, bound=[-0.51, 0.51])
        self.energy_unit = const.h**2/8

    def T(self, k):
        return 0.5*k**2 * const.hbar**2 / self.energy_unit

    def U(self, x):
        return np.array(list(map(lambda x: 1e15 if abs(x)>0.5 else 0.0, x)))

def testParticleInABox():
    q = ParticleInABox()
    E, s = q.States()
    x, = q.grid.x()
    x = np.linspace(x[0], x[-1], len(s[:,0]))
    #plt.plot(x, q.U(x))
    for i in range(10):
        plt.hlines(E[i], x[0], x[-1], color='black')
        plt.plot(x, 30*np.real(s[:,i])+E[i], color='red', alpha=0.7)
        print(E[i])
    plt.show()

class ParticleInABox2D(QuantumSystem):
    def __init__(self):
        super(ParticleInABox2D, self).__init__(dim=2)
        self.set_grid(size=[61,61], bound=[[-0.51,0.51],[-0.51, 0.51]])
        self.energy_unit = const.h**2/8

    def T(self, k1, k2):
        return 0.5*(k1**2+k2**2) * const.hbar**2 / self.energy_unit

    def U(self, x1,x2):
        rr = x1**2+x2**2
        ret = np.zeros(rr.shape)
        ret[rr>0.25] = 1e15
        return ret

    def plot_states(self, E, s, cols=3, rows=3):
        pos_lst = [[i,j] for i in range(rows) for j in range(cols)]
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, squeeze=False)
        extent = list(np.array(self.grid.bound).flatten())
        s_lim = np.abs(s[:,cols*rows]).max()
        for i, pos in enumerate(pos_lst):
            ax = axes[pos[0],pos[1]]
            [s.set_visible(False) for loc,s in ax.spines.items()]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(np.real(s[:,i]).reshape(*self.grid.size).T,
                cmap=plt.get_cmap('RdBu'), vmin=-s_lim, vmax=s_lim,
                origin='lower', extent=extent, aspect='equal')
            ax.set_xlabel('$E_{%d}=%g$'%(i, E[i]))
        plt.show()

def testParticleInABox2D():
    q = ParticleInABox2D()
    E, s = q.States()
    q.plot_states(E, s, cols=6, rows=6)

if __name__ == "__main__":
    testCavity()
    #testHarmonicOscillator2D()
    #testHydrogen()
    #testParticleInABox()
    #testParticleInABox2D()
    pass
