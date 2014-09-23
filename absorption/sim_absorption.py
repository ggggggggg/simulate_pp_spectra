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

binsize = 5
fwhm = 5
precounts5 = 4000000
abs_lens = 0.2
excitation_fraction = 0.28

energy = np.arange(7050,7350,binsize)
eipairs = {}
plt.figure()
for key in fnames.keys():
    data = np.loadtxt(fnames[key])
    if key == "it_diff":
        xiscaled = data[:,1]
        xi = np.interp(energy, data[:,0], xiscaled)
        xi_convolve = np.convolve(xi, gaussian_fwhm(len(xi), fwhm, binsize),"same")
    else:
        xiscaled = (data[:,1]-data[0,1])/(data[-1,1]-data[0,1]) # scale so xi is 0 to 1
        xi = np.interp(energy, data[:,0], xiscaled)
        xi_convolve = np.convolve(xi, gaussian_fwhm(len(xi), fwhm, binsize),"same")
        xi_convolve = np.abs(xi_convolve)
    eipairs[key] = energy, xi_convolve
    plt.plot(energy, xi_convolve, label=key)

plt.legend()
plt.xlabel("energy (eV)")
plt.ylabel("counts per %g eV bins"%binsize)

#rentzepis sim
energy_g, xi_g = eipairs["rentz_g"]
energy_p, xi_p_rentz = eipairs["rentz_p"]
xi_e = (xi_p_rentz-(1-excitation_fraction)*xi_g)/excitation_fraction
xi_p = (1-excitation_fraction)*xi_g+excitation_fraction*xi_e
avg_g = (precounts5*binsize/5.0)*np.exp(-abs_lens*xi_g)
avg_p = (precounts5*binsize/5.0)*np.exp(-abs_lens*xi_p)
counts_g = np.array([np.random.poisson(n) for n in avg_g])
counts_p = np.array([np.random.poisson(n) for n in avg_p])

plt.figure(figsize=(16,8))
plt.subplot(221)
plt.plot(energy_g, counts_g, label="unpumped")
plt.plot(energy_g, counts_p, label="pumped")
plt.ylabel("count/%g eV"%binsize)
plt.legend(loc="upper right")
plt.grid("on")
plt.title("rentzepis, fwhm=%g, precounts5=%g, abs_lens=%g"%(fwhm, precounts5, abs_lens))
plt.subplot(223)
plt.plot(energy_g, (counts_p-counts_g)/np.sqrt(counts_g+counts_p),".-")
plt.xlabel("energy (eV)")
plt.grid("on")
plt.ylabel("pumped-unpumped/sqrt(p+u)")


#japan sim
energy_g, xi_g = eipairs["japan_g"]
energy_p, xi_e = eipairs["japan_e"]
xi_p = (1-excitation_fraction)*xi_g+excitation_fraction*xi_e
avg_g = (precounts5*binsize/5.0)*np.exp(-abs_lens*xi_g)
avg_p = (precounts5*binsize/5.0)*np.exp(-abs_lens*xi_p)
counts_g = np.array([np.random.poisson(n) for n in avg_g])
counts_p = np.array([np.random.poisson(n) for n in avg_p])

plt.subplot(222)
plt.plot(energy_g, counts_g, label="unpumped")
plt.plot(energy_g, counts_p, label="pumped")
plt.ylabel("count/%g eV"%binsize)
plt.legend(loc="upper right")
plt.grid("on")
plt.title("japan, binsize=%g, excitation frac=%g"%(binsize, excitation_fraction))
plt.subplot(224)
plt.plot(energy_g, (counts_p-counts_g)/np.sqrt(counts_g+counts_p),".-")
plt.xlabel("energy (eV)")
plt.grid("on")
plt.ylabel("pumped-unpumped/sqrt(p+u)")

#iron tris sim
energy_g, xi_g = eipairs["it_g"]
energy_d, xi_d = eipairs["it_diff"]
xi_p = xi_g+excitation_fraction*xi_d
avg_g = (precounts5*binsize/5.0)*np.exp(-abs_lens*xi_g)
avg_p = (precounts5*binsize/5.0)*np.exp(-abs_lens*xi_p)
counts_g = np.array([np.random.poisson(n) for n in avg_g])
counts_p = np.array([np.random.poisson(n) for n in avg_p])

plt.figure(figsize=(8,8))
plt.subplot(211)
plt.plot(energy_g, counts_g, label="unpumped")
plt.plot(energy_g, counts_p, label="pumped")
plt.ylabel("count/%g eV"%binsize)
plt.legend(loc="upper right")
plt.grid("on")
plt.title("iron tris, binsize=%g, excitation frac=%g,\n fwhm=%g, precounts5=%g, abs_lens=%g"%(binsize, excitation_fraction, fwhm, precounts5, abs_lens))
plt.subplot(212)
plt.plot(energy_g, (counts_p-counts_g)/np.sqrt(counts_g+counts_p),".-")
plt.xlabel("energy (eV)")
plt.grid("on")
plt.ylabel("pumped-unpumped/sqrt(p+u)")

