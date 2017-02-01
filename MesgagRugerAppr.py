# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:34:12 2017

@author: GrinevskiyAS
"""
from __future__ import division
import numpy as np
from numpy import pi, sin, cos, tan
import matplotlib.pyplot as plt
from matplotlib import cm

data_input=np.loadtxt(r"E:\Aspir_data\SynModel1\aniso_model_UsedInProject.dat")


Nmodels = 100
H_layer = 200
H_between = 400
dh = 2


depth = data_input[:,0]
vp = data_input[:,1]
vs = data_input[:,2]
dn = data_input[:,3]
de = data_input[:,4]
ep = data_input[:,5]
ga = data_input[:,6]
az0 = np.zeros_like(depth, dtype = float)+90
fi0 = pi/2 - pi*az0 / 180
eta = (ep-de)/(1+2*de)
mu = dn * vs**2
vsvp = vs/vp
vpvs = vp/vs



deta = np.insert(np.diff(eta),0,0)

Nd = len(depth)

az_list = np.arange(0, 180, 22.5)
fi_list = pi/2 - pi*az_list / 180

ang_list = np.arange(0, 50, 5)
th_list = pi*ang_list/180

starttime = 0
dt = 2

time = np.cumsum(2000*dh/vp)
time_fl = dt*np.floor(time/dt)
ind_d_top = np.floor(((np.arange(Nmodels))*(H_between + H_layer) + H_between)/dh).astype(int)
ind_d_bot = np.floor(((np.arange(Nmodels)+1)*(H_between + H_layer))/dh).astype(int)

times_top = time[ind_d_top]
times_bot = time[ind_d_bot]


ind_t_top = np.floor((times_top - starttime)/dt).astype(int)
ind_t_bot = np.floor((times_bot - starttime)/dt).astype(int)


ind1 = ind_d_top[1]


def ReflCoef(q):
    rc = 0.5 * np.diff(q) / np.mean(np.row_stack((q[1:], q[:-1])), axis = 0)
    return np.hstack((0, rc))

    

   
    
r_zp = ReflCoef(vp*dn)

def ComputeRugerReflection(vp, vs, dn, de, ep, ga, fi0, fi_list, th_list):
    res = np.zeros((len(vp), len(fi_list), len(th_list)), dtype = float)

    r0 = ReflCoef(vp * dn)
    
    mu = dn * vs**2
    
    MnVp = np.hstack( (vp[0], np.mean(np.row_stack((vp[1:], vp[:-1])), axis = 0)) )
    MnVs = np.hstack( (vs[0], np.mean(np.row_stack((vs[1:], vs[:-1])), axis = 0)) )
    dde = np.insert(np.diff(de), 0, 0)
    dep = np.insert(np.diff(ep), 0, 0)
    ga_vti = -ga/(1 + 2*ga)
    dga_vti = np.insert(np.diff(ga_vti), 0, 0)
    
    #слагаемые для r2
    part1 = 2 * ReflCoef(vp)
    part2 = (2*MnVs/MnVp)**2 * (2*ReflCoef(mu))
    part3 = dde + 8*(MnVs/MnVp)**2 * dga_vti


    for ifi, fi in enumerate(fi_list):
        r2 = 0.5 * (part1 - part2 + part3 * cos(fi - fi0)**2)
        r4 = 0.5 * (2 *ReflCoef(vp) + dep * cos(fi-fi0)**4 + dde * sin(fi-fi0)**2 * cos(fi-fi0)**2)
        
        for ith, th in enumerate(th_list):
            resij = r0 + r2*sin(th)**2 + r4 * sin(th)**2 * tan(th)**2
            res[:, ifi, ith] = resij
        
    return res

rugeramp = ComputeRugerReflection(vp, vs, dn, de, ep, ga, fi0, fi_list, th_list)

def ComputeMesdagReflection(vp, vs, dn, de, ep, ga, fi0, fi_list, th_list):
    res = np.zeros((len(r_zp), len(fi_list), len(th_list)), dtype = float)
    mn_de = np.hstack( (de[0], np.mean(np.row_stack((de[1:], de[:-1])), axis = 0)) )   
    mn_ep = np.hstack( (ep[0], np.mean(np.row_stack((ep[1:], ep[:-1])), axis = 0)) )       
    mn_ga = np.hstack( (ga[0], np.mean(np.row_stack((ga[1:], ga[:-1])), axis = 0)) )   
    der = (de + 1 - mn_de) / (1 - mn_de)
    epr = (ep + 1 - mn_ep) / (1 - mn_ep)
    gar = (ga + 1 - mn_ga) / (1 - mn_ga)
    K = (vs/vp)**2
    Kcoef = (4*K+1)/(8*K)
    
    for ifi, fi in enumerate(fi_list):
        cos2az = cos(fi - fi0)**2
        vp_az = vp * der**cos2az  * (epr/der)**(cos2az**2)
        vs_az = vs * (np.sqrt(der)/gar)**cos2az * (epr/der)**(Kcoef*cos2az**2)
        dn_az = dn * der**(-cos2az) * (epr/der)**(-cos2az**2)
            
        for ith, th in enumerate(th_list):
            r0 = ReflCoef(vp_az) + ReflCoef(dn_az)
            r2 = ReflCoef(vp_az) - 2 * (vs_az/vp_az)**2 * (4*ReflCoef(vp_az) + 2*ReflCoef(dn_az))
            r4 = ReflCoef(vp_az)
            
            
            resij = r0 + r2 * sin(th)**2 + r4 * (tan(th)**2 - sin(th)**2)
            res[:, ifi, ith] = resij

    return res

mesdagamp = ComputeMesdagReflection(vp, vs, dn, de, ep, ga, fi0, fi_list, th_list)

def PlotRugerAmp(ax, amp, ind, ang_list, az_list, cmap = cm.Spectral_r, vid = 'Az'):
    N_ang = len(ang_list)
    if vid == 'Az':
        data_plot = amp[ind, :, :].T
        az_list_plot = az_list
        if not (abs(az_list[-1] - az_list[0]) == 180):
            az_list_plot = np.hstack((az_list, az_list[0] + 180))
            data_plot = np.column_stack((data_plot, data_plot[:,0]))
    
        cm_subsection = np.linspace(0.0,1.0, N_ang)
        colors = [ cmap(x) for x in cm_subsection ]
        for i, ang in enumerate(ang_list):
            ax.plot(az_list_plot, data_plot[i,:], marker = 'o', markerfacecolor=colors[i], markersize = 9, markeredgecolor = 'None',
                    linewidth = 0.5, color = colors[i], label = str(int(ang)))
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::1], labels[::1], ncol=int(N_ang/2), loc='best', prop = {'size': 12})
        
    elif vid == 'An':
        data_plot = amp[ind, :, :]
        cm_subsection = np.linspace(0.0,1.0, N_ang)
        colors = [ cmap(x) for x in cm_subsection ]
        for i, azi in enumerate(az_list):
            ax.plot(ang_list, data_plot[i,:], marker = 'o', markerfacecolor=colors[i], markersize = 6, markeredgecolor = 'None',
                    linewidth = 1.5, color = colors[i], label = str(azi))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='best',  prop = {'size': 12}, framealpha=0.3)
    

fgr = plt.figure(facecolor= 'white', figsize = [18,10])
ax_an = fgr.add_subplot(121)
ax_az = fgr.add_subplot(122)
fgr.canvas.set_window_title('Ruger, computed')

PlotRugerAmp(ax_az, rugeramp, ind_d_top[4], ang_list, az_list, cmap = cm.Accent, vid = 'Az')
PlotRugerAmp(ax_an, rugeramp, ind_d_top[4], ang_list, az_list, cmap = cm.Accent, vid = 'An')

fgr.tight_layout()


fgr_akir = plt.figure(facecolor= 'white', figsize = [18,10])
ax_an_akir = fgr_akir.add_subplot(121)
ax_az_akir = fgr_akir.add_subplot(122)
fgr_akir.canvas.set_window_title('Aki-Richards + Mesdag, computed')

PlotRugerAmp(ax_az_akir, mesdagamp, ind_d_top[4], ang_list, az_list, cmap = cm.Accent, vid = 'Az')
PlotRugerAmp(ax_an_akir, mesdagamp, ind_d_top[4], ang_list, az_list, cmap = cm.Accent, vid = 'An')
