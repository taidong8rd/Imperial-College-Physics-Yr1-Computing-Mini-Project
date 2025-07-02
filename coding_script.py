# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:22:05 2019

@author: sz3119
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob as glob

xe = 656.28
c = 2.9979e8

file_list = glob.glob('Data/Halpha_spectral_data/*.csv')

V0 = []
Distance = []

for i in range(len(file_list)):
    file_name = file_list[i]
    with open(file_name,'r') as file:
        line1 = file.readline()
        line1_split = line1.split(',')
        
        str1 = line1_split[3].strip()     
        str2 = line1_split[2].strip()     
        
        if str1 == 'Instrument Response: Good':
            ob = int(str2.strip('Observation:'))
            
            wavelength,intensity = np.loadtxt(file_list[i],skiprows=2,unpack=1,delimiter=',')

            fit_intensity,cov_intensity = np.polyfit(wavelength,intensity,1,cov=1)
            pintensity = np.poly1d(fit_intensity)
            residual = intensity - pintensity(wavelength)
           
            guess_a = max(residual)    
            
            index = np.argmax(residual) 
            guess_mu = wavelength[index]
            
            guess_m = fit_intensity[0]
            guess_c = fit_intensity[1]  
                    
            def fit_func(wavelength,a,mu,sig,m,c):
                gaus = a*np.exp(-(wavelength-mu)**2/(2*sig**2))
                line = m*wavelength+c
                return gaus + line    
                
            initial_guess = [guess_a,guess_mu,10,guess_m,guess_c]
            po,po_cov = curve_fit(fit_func,wavelength,intensity,initial_guess)
            
            def v_func(x0):
                V = c*(x0**2-xe**2)/(x0**2+xe**2)
                return V
            
            v0 = v_func(po[1])/1000          
            V0.append(v0)
            
            observation_number,distance = np.loadtxt('Data/Distance_Mpc.csv',unpack=1,skiprows=1,delimiter=',')
            
            for j in range(len(observation_number)):
                if ob == observation_number[j]:
                    Distance.append(distance[j])
    
fit_V0,cov_V0 = np.polyfit(Distance,V0,1,cov=1)
pV0 = np.poly1d(fit_V0)

plt.grid()
plt.xlabel('Distance (Mpc)')
plt.ylabel('Redshift Velocity (km/s)')
plt.title('Plot of Redshift Velocity vs. Distance')
plt.plot(Distance,V0,'x')
plt.plot(Distance,pV0(Distance))
#plt.savefig('Data/Plot of Redshift Velocity vs. Distance.png')
plt.show()

print('H0 =',fit_V0[0],'(km/s)/Mpc')
print('Error = +-',np.sqrt(cov_V0[0,0]),'(km/s)/Mpc')
