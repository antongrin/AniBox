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
import scipy.linalg as la


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


def GenRicker(f, length = 200, dt = 2):
    length = length / 1000
    dt = dt/1000
    t = np.arange(-length/2, (length+dt)/2, dt)
    y = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2))
    return t, y

def TransformToTime(vp, times, t, mode = 'mean'):
    #mode can be mean, median, nearest    
    out = np.zeros_like(t)*np.nan
    out[0] = vp[np.argmin(abs(times-t[0]))]
    out[-1] = vp[np.argmin(abs(times-t[1]))]
    for i, ti in enumerate(t):
        if i>0 and i<len(t)-1:        
            tmin = 0.5*(t[i-1] + t[i])
            tmax = 0.5*(t[i+1] + t[i])
            indi = (times>=tmin) & (times<tmax)

            if mode == 'mean':
                out[i] = np.mean(vp[indi])
            elif mode == 'median':
                out[i] = np.median(vp[indi])
            elif mode == 'nearest':      
                indmid = np.argmin(abs(times-ti))
                out[i] = vp[indmid]
            
    return out

def GenerateSynData(r, wav):
    out = np.zeros_like(r)
    if r.ndim == 2:
        for i in xrange(np.shape(out)[1]):
            out[:,i] = np.convolve(r[:,i], wav, mode = 'same')
    elif r.ndim == 1:
            out = np.convolve(r, wav, mode = 'same')

    return out


def AVAzRugerInv(R, th_list, fi_list):
    #R сортирована по углу и азимуту
    R = R.ravel()    
    
    th_sequence = np.tile(th_list, (len(fi_list), 1)).ravel(1)
    fi_sequence = np.tile(fi_list, (len(th_list), 1)).ravel()
    
    #считаем синусы и тангенсы углов падения
    s2ang=sin(th_sequence)**2
    
    #строим матрицу G и матрицу наблюденных амплитуд R
    G1 = np.ones(len(th_sequence), dtype = float)
    G2 = s2ang
    G3 = cos(2*fi_sequence)*s2ang
    G4 = sin(2*fi_sequence)*s2ang
    M = np.column_stack((G1, G2, G3, G4))
    
    #методом МНК определяем параметры аппроксимации
#    res = np.linalg.lstsq(M, R)
#    A=res[0][0][0]
#    B=res[0][1][0]
#    C=res[0][2][0]
#    D=res[0][3][0]
    
    lam=1e-1
    res=la.inv(M.T.dot(M)+lam*np.eye(4)).dot(M.T).dot(R)
    A=res[0]
    B=res[1]
    C=res[2]
    D=res[3]
    
    #    
#    Az0 = np.arctan(D/C)/2
#    Az0 = 180*0.5*np.arctan2(D,C)/pi
    Az0 = 180*0.5*np.arctan(D/C)/pi
    Bani = 2*np.sqrt(C**2 + D**2)
    Biso = B - 0.5*Bani
    Bani2 = -2*np.sqrt(C**2 + D**2)
    Biso2 = B - 0.5*Bani2

    
    #считаем аппроксимирующую кривую и погрешность
    appr = A + (Biso + Bani*sin(fi_sequence - 90*Az0/pi)**2)*sin(th_sequence)**2
    appr2 = A + (Biso2 + Bani2*sin(fi_sequence - 90*Az0/pi)**2)*sin(th_sequence)**2

    err = np.sum((R - appr)**2)
    err2 = np.sum((R - appr2)**2)
    
    if (err2>err):
        return A, Biso, Bani, Az0, err
    else:
        return A, Biso2, Bani2, Az0, err2


def RugerApprTest(syn_data, A_inv, Biso_inv, Bani_inv, Az0_inv, ind_test, ang_list, az_list):
    f = plt.figure(facecolor= 'white', figsize = [14,7])
    ax_an = f.add_subplot(121)
    PlotRugerAmp(ax_an, syn_data[ind_test].reshape( (1, len(az_list), len(ang_list) ), order = 'F'), 0, ang_list, az_list, cmap = cm.Accent, vid = 'An')
    ax_az = f.add_subplot(122)
    PlotRugerAmp(ax_az, syn_data[ind_test].reshape( (1, len(az_list), len(ang_list) ), order = 'F'), 0, ang_list, az_list, cmap = cm.Accent, vid = 'Az')
    f.tight_layout()
    
    for line in ax_an.lines:
        line.set_linewidth(0)
    for line in ax_az.lines:
        line.set_linewidth(0)


    cm_subsection = np.linspace(0.0, 1.0, len(az_list))
    colors = [ cm.Accent(x) for x in cm_subsection ]    
    
    for iaz, az in enumerate(az_list):

        amp_appr = A_inv[ind_test] + (Biso_inv[ind_test] + Bani_inv[ind_test]*sin(pi*(az - Az0_inv[ind_test])/180)**2)*sin(pi*ang_list/180)**2
        ax_an.plot(ang_list, amp_appr, c = colors[iaz])

    cm_subsection = np.linspace(0.0, 1.0, len(ang_list))
    colors = [ cm.Accent(x) for x in cm_subsection ]    
    for ian, an in enumerate(ang_list):

        amp_appr = A_inv[ind_test] + (Biso_inv[ind_test] + Bani_inv[ind_test]*sin(pi*(az_list - Az0_inv[ind_test])/180)**2)*sin(pi*an/180)**2
        ax_az.plot(az_list, amp_appr, c = colors[ian])
        
    
    
#    print 'error = ', comp_error


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

mdl_name = ['A', 'B', 'C', 'D']
de_pl = np.array([0,   -0.1,   0,    -0.05])
ep_pl = np.array([0,    0,    -0.1,  -0.05])
#ga_pl = np.array([-0.1,  0,     0,   -0.15])
#ga_vti_pl = -ga_pl/(1 + 2*ga_pl)
ga_vti_pl = np.array([0.1,  0,     0,   0.15])
ga_pl = -ga_vti_pl/(1 + 2*ga_vti_pl)
az0_pl = np.array([0,0,0,0])

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


az_list = np.arange(0, 180, 22.5)
fi_list = pi*az_list / 180
ang_list = np.arange(0, 45 + 0.1, 5)
th_list = pi*ang_list/180


starttime = 0
dt = 2


time = np.cumsum(2000*dh/vp)
time_fl = dt*np.floor(time/dt)
ind_d_top = np.floor(((np.arange(Nmodels))*(H_between + H_layer) + H_between)/dh).astype(int)
ind_d_bot = np.floor(((np.arange(Nmodels)+1)*(H_between + H_layer))/dh).astype(int)
times_top = time[ind_d_top]
times_bot = time[ind_d_bot]

ind_t_top = np.round((times_top - starttime)/dt).astype(int)
ind_t_bot = np.round((times_bot - starttime)/dt).astype(int)

t = np.arange(starttime, max(time), dt)

Model_no = 1


vp_t = TransformToTime(vp, time, t, mode = 'nearest')
vs_t = TransformToTime(vs, time, t, mode = 'nearest')
dn_t = TransformToTime(dn, time, t, mode = 'nearest')
de_t = TransformToTime(de, time, t, mode = 'nearest')
ep_t = TransformToTime(ep, time, t, mode = 'nearest')
ga_t = TransformToTime(ga, time, t, mode = 'nearest')
fi0_t = TransformToTime(fi0, time, t, mode = 'nearest')




rugeramp = ComputeRugerReflection(vp, vs, dn, de, ep, ga, fi0, fi_list, th_list)
rugeramp_t = ComputeRugerReflection(vp_t, vs_t, dn_t, de_t, ep_t, ga_t, fi0_t, fi_list, th_list)


rugeramp_cang_caz = rugeramp_t.reshape((np.shape(rugeramp_t)[0], np.shape(rugeramp_t)[1]*np.shape(rugeramp_t)[2]), order = 'F').copy()
rugeramp_caz_cang = rugeramp_t.reshape((np.shape(rugeramp_t)[0], np.shape(rugeramp_t)[1]*np.shape(rugeramp_t)[2]), order = 'C').copy()



f = 40
t_wav, wav = GenRicker(f, length = 200, dt = dt)

syn_data = GenerateSynData(rugeramp_cang_caz, wav)

#plt.imshow(syn_data, cmap = 'Greys', aspect = 'auto', interpolation = 'none')
#
#
#fgr = plt.figure(facecolor= 'white', figsize = [14,7])
#ax_an = fgr.add_subplot(121)
#ax_az = fgr.add_subplot(122)
#fgr.canvas.set_window_title('Ruger, modelled amplitudes for model {0} (time = {1})'.format(mdl_name[Model_no], t[ind_t_top[Model_no]]))
#
#PlotRugerAmp(ax_az, syn_data.reshape((np.shape(syn_data)[0], np.shape(rugeramp)[1],np.shape(rugeramp)[2]), order = 'F'), ind_t_top[Model_no]+1, ang_list, az_list, cmap = cm.Accent, vid = 'Az')
#PlotRugerAmp(ax_an, syn_data.reshape((np.shape(syn_data)[0], np.shape(rugeramp)[1],np.shape(rugeramp)[2]), order = 'F'), ind_t_top[Model_no]+1, ang_list, az_list, cmap = cm.Accent, vid = 'An')
#
#fgr.tight_layout()
#
#for ax in [ax_an, ax_az]:
#    ax.set_ylim([0, 0.1])



#inversion for avaz parameters

minang_avoaz = 0
maxang_avoaz = 30
th_avoaz = th_list[(th_list >= pi*minang_avoaz/180)&(th_list <= pi*maxang_avoaz/180)]

th_sequence = np.tile(th_list, (len(fi_list), 1)).ravel(1)
fi_sequence = np.tile(fi_list, (len(th_list), 1)).ravel()

ind_seq_for_avoaz = (th_sequence >= pi*minang_avoaz/180)&(th_sequence <= pi*maxang_avoaz/180)
th_sequence_avo = th_sequence[ind_seq_for_avoaz]
fi_sequence_avo = fi_sequence[ind_seq_for_avoaz]
syn_data_for_avoaz = syn_data[:, ind_seq_for_avoaz]

A_inv = np.zeros_like(t, dtype = float)
Biso_inv = np.zeros_like(t, dtype = float)
Bani_inv = np.zeros_like(t, dtype = float)
Az0_inv = np.zeros_like(t, dtype = float)
err_inv = np.zeros_like(t, dtype = float)
err2_inv = np.zeros_like(t, dtype = float)

for i in xrange(len(t)):
    R = syn_data_for_avoaz[i, :].reshape((len(th_sequence_avo), 1)) #делаем вектор строку
    [A_i, Biso_i, Bani_i, Az0_i, err_i] = AVAzRugerInv(R, th_avoaz, fi_list)
    A_inv[i] = A_i
    Biso_inv[i] = Biso_i
    Bani_inv[i] = Bani_i
    Az0_inv[i] = Az0_i
    err_inv[i]=err_i

#    nearstkall[k,i]=np.sum(data[k,i,0:3])
    
f_avoaz = plt.figure(facecolor= 'white', figsize = [14,7])
ax_a = f_avoaz.add_subplot(411)
ax_b = f_avoaz.add_subplot(412, sharex = ax_a)
ax_bani = f_avoaz.add_subplot(413, sharex = ax_a)
ax_az = f_avoaz.add_subplot(414, sharex = ax_a)
ax_a.plot(t, A_inv)
ax_a.plot(time, ReflCoef(vp*dn))
ax_b.plot(t, Biso_inv)
B_theor = 0.5 * (2*ReflCoef(vp) - (2*vs/vp)**2 * 2*ReflCoef(dn * vs**2))
ax_b.plot(time, B_theor)
#ax_b.plot(t, Biso_inv)
ax_bani.plot(t, Bani_inv)
#ax_bani.plot(t, np.cumsum(Bani_inv))
#ax_bani.plot(t, np.cumsum(np.cumsum(Bani_inv))/np.sum(np.cumsum(np.cumsum(wav))))


ax_az.plot(t, err_inv)
ax_az.plot(t, err2_inv)

MnVp = np.hstack( (vp[0], np.mean(np.row_stack((vp[1:], vp[:-1])), axis = 0)) )
MnVs = np.hstack( (vs[0], np.mean(np.row_stack((vs[1:], vs[:-1])), axis = 0)) )
dde = np.insert(np.diff(de), 0, 0)
dep = np.insert(np.diff(ep), 0, 0)
ga_vti = -ga/(1 + 2*ga)
dga_vti = np.insert(np.diff(ga_vti), 0, 0)

Bani_theor = 0.5*(dde + 8*(MnVs/MnVp)**2 * dga_vti)
ax_bani.plot(time, Bani_theor)
ax_bani.plot(t, np.cumsum(np.cumsum(np.cumsum(Bani_inv)))/ np.sum(np.cumsum(np.cumsum(wav))))
f_avoaz.tight_layout()


#t_test = t[ind_t_top[0]]
t_test = 515
ind_test = np.where(t >= t_test)[0][0]

RugerApprTest(syn_data, A_inv, Biso_inv, Bani_inv, Az0_inv, ind_test, ang_list, az_list)
#RugerApprTest(syn_data, A_inv, Biso_inv+Bani_inv, -Bani_inv, Az0_inv, ind_test, ang_list, az_list)



