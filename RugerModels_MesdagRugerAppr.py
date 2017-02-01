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


def PlotModel(depth, vp, vs, dn, ep, de, ga):
    f = plt.figure(figsize = (12,10), facecolor = 'w')
    ax_vp = f.add_subplot(161)
    ax_vs = f.add_subplot(162)
    ax_dn = f.add_subplot(163)
    ax_de = f.add_subplot(164)
    ax_ep = f.add_subplot(165)
    ax_ga = f.add_subplot(166)
    
    data = np.column_stack((vp, vs, dn, ep, de, ga))
    names = ('vp', 'vs', 'dn', 'ep', 'de', 'ga')
    for i, ax in enumerate([ax_vp, ax_vs, ax_dn, ax_de, ax_ep, ax_ga]):
        ax.plot(data[:,i], depth, lw = 1.5)
        ax.invert_yaxis()
        ax.set_xlabel(names[i])
        ax.xaxis.set_ticks_position('top')
    f.tight_layout()

def ReflCoef(q):
    rc = 0.5 * np.diff(q) / np.mean(np.row_stack((q[1:], q[:-1])), axis = 0)
    return np.hstack((0, rc))


def ComputeRugerReflection(vp, vs, dn, de, ep, ga, fi0, fi_list, th_list):
    # уравнение взято из Mesdag, но у него азимуты относятся к медленным волнам
    # поэтому все синусы заменил на косинусы и наоборот
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
        r2 = 0.5 * (part1 - part2 + part3 * sin(fi - fi0)**2)
        r4 = 0.5 * (2 * ReflCoef(vp) + dep * sin(fi-fi0)**4 + dde * cos(fi-fi0)**2 * sin(fi-fi0)**2)
        
        for ith, th in enumerate(th_list):
            resij = r0 + r2*sin(th)**2 + r4 * sin(th)**2 * tan(th)**2
            res[:, ifi, ith] = resij
        
    return res

def ComputeMesdagReflection(vp, vs, dn, de, ep, ga, fi0, fi_list, th_list):
    # у него все формулы выражены в азимутах медленной, а не быстрой волны
    # поэтому в формуле для cos2az добавляем 90 градусов 
    # уравнение Аки-Ричардса взято из "Rock-physics relationships between inverted elastic reflectivities"
    # и вроде оно корректно, судя по тестам с нулевой анизотропией
    
    
    res = np.zeros((len(vp), len(fi_list), len(th_list)), dtype = float)
    ga = -ga/(1 + 2*ga)
    mn_de = np.hstack( (de[0], np.mean(np.row_stack((de[1:], de[:-1])), axis = 0)) )
    mn_ep = np.hstack( (ep[0], np.mean(np.row_stack((ep[1:], ep[:-1])), axis = 0)) )
    mn_ga = np.hstack( (ga[0], np.mean(np.row_stack((ga[1:], ga[:-1])), axis = 0)) )
    der = (de + 1 - mn_de) / (1 - mn_de)
    epr = (ep + 1 - mn_ep) / (1 - mn_ep)
    gar = (ga + 1 - mn_ga) / (1 - mn_ga)
#    der = (de + 1 - mn_de)
#    epr = (ep + 1 - mn_ep)
#    gar = (ga + 1 - mn_ga)

    K = (vs/vp)**2
    Kcoef = (4*K+1)/(8*K)
    
    for ifi, fi in enumerate(fi_list):
        cos2az = cos(fi - fi0 + pi/2)**2
        vp_az = vp * der**cos2az  * (epr/der)**(cos2az**2)
        vs_az = vs * (np.sqrt(der)/gar)**cos2az * (epr/der)**(Kcoef*cos2az**2)
        dn_az = dn * der**(-cos2az) * (epr/der)**(-cos2az**2)

        for ith, th in enumerate(th_list):
            r0 = ReflCoef(vp_az*dn_az)
            r2 = 0.5 * (2*ReflCoef(vp_az) - (2*vs_az/vp_az)**2 * 2*ReflCoef(dn_az * vs_az**2))
            r4 = ReflCoef(vp_az)
            
            
            resij = r0 + r2 * sin(th)**2 + r4 * (tan(th)**2 - sin(th)**2)
            res[:, ifi, ith] = resij

    return res

def PlotRugerAmp(ax, amp, ind, ang_list, az_list, cmap = cm.Spectral_r, vid = 'Az'):
    N_ang = len(ang_list)
    N_az = len(az_list)
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
        cm_subsection = np.linspace(0.0,1.0, N_az)
        colors = [ cmap(x) for x in cm_subsection ]
        for i, azi in enumerate(az_list):
            ax.plot(ang_list, data_plot[i,:], marker = 'o', markerfacecolor=colors[i], markersize = 6, markeredgecolor = 'None',
                    linewidth = 1.5, color = colors[i], label = str(azi))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='best',  prop = {'size': 12}, framealpha=0.3)



H_layer = 200
H_between = 400
dh = 2


#параметры пласта
vp_pl = 2500.0
vs_pl = 1500.0
dn_pl = 2.7
vpvs_pl = vp_pl/vs_pl
zp_pl = vp_pl * dn_pl
mu_pl = dn_pl * vs_pl**2

#параметры вмещающих
vp_vm = vp_pl * (2 - 0.1)/(2 + 0.1)
zp_vm = zp_pl * (2 - 0.1)/(2 + 0.1)
dn_vm = zp_vm / vp_vm
mu_vm = mu_pl * (2 - 0.2)/(2 + 0.2)
vs_vm = np.sqrt(mu_vm / dn_vm)
vpvs_vm = vp_vm/vs_vm


de_pl = np.array([0,   -0.1,   0,    -0.05])
ep_pl = np.array([0,    0,    -0.1,  -0.05])
#ga_pl = np.array([-0.1,  0,     0,   -0.15])

#ga_vti_pl = -ga_pl/(1 + 2*ga_pl)
ga_vti_pl = np.array([0.1,  0,     0,   0.15])
ga_pl = -ga_vti_pl/(1 + 2*ga_vti_pl)
az0_pl = np.array([0,0,0,0])-90

Nmodels = len(de_pl)

depth = np.arange(dh, (H_layer + H_between)*Nmodels + H_between, dh)
vp = np.ones_like(depth).astype(float) * vp_vm
vs = np.ones_like(depth).astype(float) * vs_vm
dn = np.ones_like(depth).astype(float) * dn_vm
ep = np.zeros_like(depth).astype(float)
de = np.zeros_like(depth).astype(float)
ga = np.zeros_like(depth).astype(float)
az0 = np.zeros_like(depth).astype(float)


for i in xrange(Nmodels):
    i_start = np.floor((i*(H_between + H_layer) + H_between)/dh).astype(int)
    i_end = np.floor((i+1)*(H_between + H_layer)/dh).astype(int)
    
    vp[i_start:i_end] = vp_pl
    vs[i_start:i_end] = vs_pl
    dn[i_start:i_end] = dn_pl
    
    ep[i_start:i_end] = ep_pl[i]
    de[i_start:i_end] = de_pl[i]
    ga[i_start:i_end] = ga_pl[i]
    az0[i_start:i_end] = az0_pl[i]

zp = vp*dn
mu = dn*vs**2

fi0 = pi*az0/180

#PlotModel(depth, vp, vs, dn, ep, de, ga)


az_list = np.arange(0, 180, 5)
fi_list = pi*az_list / 180
ang_list = np.arange(0, 45 + 0.1, 1)
th_list = pi*ang_list/180

az_list_sparse = np.arange(0, 91, 30)
fi_list_sparse = pi*az_list_sparse / 180
ang_list_sparse = np.arange(10, 41, 10)
th_list_sparse = pi*ang_list_sparse/180

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


Model_no = 3


rugeramp_an = ComputeRugerReflection(vp, vs, dn, de, ep, ga, fi0, fi_list_sparse, th_list)
mesdagamp_an = ComputeMesdagReflection(vp, vs, dn, de, ep, ga, fi0, fi_list_sparse, th_list)
rugeramp_az = ComputeRugerReflection(vp, vs, dn, de, ep, ga, fi0, fi_list, th_list_sparse)
mesdagamp_az = ComputeMesdagReflection(vp, vs, dn, de, ep, ga, fi0, fi_list, th_list_sparse)

fgr = plt.figure(facecolor= 'white', figsize = [14,7])
ax_an = fgr.add_subplot(121)
ax_az = fgr.add_subplot(122)
fgr.canvas.set_window_title('Ruger, computed')

PlotRugerAmp(ax_az, rugeramp_az, ind_d_top[Model_no], ang_list_sparse, az_list, cmap = cm.Accent, vid = 'Az')
PlotRugerAmp(ax_an, rugeramp_an, ind_d_top[Model_no], ang_list, az_list_sparse, cmap = cm.Accent, vid = 'An')

fgr.tight_layout()

fgr_akir = plt.figure(facecolor= 'white', figsize = [14,7])
ax_an_akir = fgr_akir.add_subplot(121)
ax_az_akir = fgr_akir.add_subplot(122)
fgr_akir.canvas.set_window_title('Aki-Richards + Mesdag, computed')

PlotRugerAmp(ax_az_akir, mesdagamp_az, ind_d_top[Model_no], ang_list_sparse, az_list, cmap = cm.Accent, vid = 'Az')
PlotRugerAmp(ax_an_akir, mesdagamp_an, ind_d_top[Model_no], ang_list, az_list_sparse, cmap = cm.Accent, vid = 'An')

fgr_akir.tight_layout()

for ax in [ax_an, ax_an_akir, ax_az, ax_az_akir]:
    ax.set_ylim([0, 0.1])

