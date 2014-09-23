import numpy as np
import pylab as plt
import scipy
import scipy.signal

fnames = {"rentz_g":"rentzepis_ground.txt", "rentz_p":"rentzepis_pumped.txt", "japan_g":"sacla_ground.txt",
          "japan_e":"sacla_excited.txt","it_g":"iron_tris_science_ground.txt",
          "it_diff":"iron_tris_science_p_minus_g.txt", "canada_g":"canada_ferrioxalate.mes"}

def gauss(x, mean, std):
    return np.exp(-((x-mean)**2/(2*std**2)))/np.sqrt(2*np.pi*std**2)

def gauss_fwhm(x,mean, fhwm):
    return gauss(x,mean, fhwm/2.355)

def gaussian_fwhm(n, fwhm, binsize):
    std =  fwhm/(2.355*binsize)
    return scipy.signal.gaussian(n, std)/np.sqrt(2*np.pi*std**2)

binsize = .5
fwhm = 5
precounts5 = 4000000
abs_lens = 0.2
excitation_fraction = 0.28

energy = np.arange(7050,7350,binsize)
eipairs = {}
key="japan_g"
data = np.loadtxt(fnames[key])
xiscaled = (data[:,1]-data[0,1])/(data[-1,1]-data[0,1]) # scale so xi is 0 to 1
xi = np.interp(energy, data[:,0], xiscaled)

plt.figure()
for fwhm in [5,8,12,16,20]:
    xi_convolve = np.convolve(xi, gaussian_fwhm(len(xi), fwhm, binsize),"same")
    # xi_convolve = np.abs(xi_convolve)
    plt.plot(energy, xi_convolve, label="%g eV"%fwhm)
plt.xlabel("energy (eV)")
plt.ylabel("xi")
plt.title(key)
plt.grid("on")
plt.legend(loc="lower right")
plt.xlim(7080, 7250)