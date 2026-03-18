"""
pkues_setup.py
Converted from pkues_setup.m
18-10-05 08:00 Hua-sheng XIE, huashengxie@gmail.com, FRI-ENN, China
Ackn.: Richard Denton (Dartmouth), Xin Tao (USTC), Jin-song Zhao (PMO), etc.
Setup for run pdrk
18-10-13 09:52 update to with loss-cone distribution
"""

import numpy as np


def pkues_setup():
    """Setup initial parameters for PDRK run. Returns a dict of all parameters."""

    config = {}

    config['savepath'] = '../output/'  # choose where to save the figures and outputs
    config['iem'] = 1       # =1, electromagnetic run; =0, electrostatic run

    config['N'] = 3          # Number harmonics, enlarge N to make sure results are convergent
    config['J'] = 8          # J-pole, usually J=8 is sufficient; other choice: 4, 12
    config['sp'] = 0         # sparse eigs(), sp=1; or eig() for all solution, sp=0
    config['wg0'] = -0.01 - 0.01j  # initial guess for sp=1, normalized by omega_{c1}

    config['B0'] = 1.21e-7   # magnetic field (Tesla)

    # Initialize the parameters k, theta, kz, kx
    par = np.zeros(6)
    par[0] = 0.01            # k, =sqrt(kx^2+kz^2), normalized by *c/omega_{p1}
    par[1] = 20              # theta, angle between k and B0, normalized by *pi/180
    par[2] = np.cos(par[1] * np.pi / 180) * par[0]  # kz, i.e., kpara*c/omega_{p1}
    par[3] = np.sin(par[1] * np.pi / 180) * par[0]  # kx, i.e., kperp*c/omega_{p1}
    config['par'] = par

    # Choose which parameter(s) to scan
    # if ipa==ipb, do 1D scan; otherwise, do 2D scan.
    # 1: k; 2: theta; 3: kz; 4: kx; 5: others.
    # Typical cases of (ipa,ipb):
    #   1. (1,1) scan k, fixed theta
    #   2. (2,2) scan theta, fixed k
    #   3. (1,2) scan 2D (k, theta)
    #   4. (3,3) scan kz, fixed kx
    #   5. (4,4) scan kx, fixed kz
    #   6. (3,4) scan 2D (kz,kx)
    config['ipa'] = 1
    config['ipb'] = 1

    config['iloga'] = 0      # ilog=0, linear scale; =1, log scale, i.e., 10^(p1:dp:p2)
    config['ilogb'] = 0

    config['pa1'] = 0.02
    config['pa2'] = 1.5
    config['dpa'] = 0.005    # 1st parameter a, depends on ipa
    config['pb1'] = 0
    config['pb2'] = 90
    config['dpb'] = 5        # 2nd parameter b, depends on ipb

    # Whether calculate polarizations (dEx,dEy,dEz,dBx,dBy,dBz) for select omega
    config['iout'] = 2       # =1, only (omega,gamma); =2, also calculate (E,B)

    # Added by Xingyu Zhu 2020-06-03
    # Whether calculate distribution function (f0+deltaf, deltaf)
    config['idf'] = 0        # =1, calc; =0, ignore
    config['jpa_df'] = 137
    config['jpb_df'] = 1
    config['jpl_df'] = 1
    config['s_df'] = 1

    if config['idf'] == 1:
        s_df = config['s_df']
        if s_df == 1:
            config['ampl'] = 0.001
            config['vxrange'] = [-3, 3]    # in vA unit
            config['vyrange'] = [-3, 3]
            config['vzrange'] = [-3, 3]
            config['vxsteps'] = 80
            config['vysteps'] = 80
            config['vzsteps'] = 80
        else:
            config['ampl'] = 0.110
            config['vxrange'] = [-100, 100]
            config['vyrange'] = [-100, 100]
            config['vzrange'] = [-100, 100]
            config['vxsteps'] = 50
            config['vysteps'] = 50
            config['vzsteps'] = 50
        config['damping'] = False
        config['const_r'] = True
        config['periods'] = False
        config['num_periods'] = 1
        config['timesteps'] = 200

    return config


if __name__ == '__main__':
    cfg = pkues_setup()
    for key, val in cfg.items():
        print(f"{key} = {val}")

