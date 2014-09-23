import numpy as np
import pylab as plt
import scipy
import scipy.signal

fnames = {"ka_hs":"fe_ka_high_spin_scrap.txt", "ka_ls":"fe_ka_low_spin_scrap.txt", "kb_ls":"fe_kb_low_spin_scrap.txt",
          "kb_hs":"fe_kb_high_spin_scrap.txt"}

def gauss(x, mean, std):
    return np.exp(-((x-mean)**2/(2*std**2)))/np.sqrt(2*np.pi*std**2)

def gauss_fwhm(x,mean, fhwm):
    return gauss(x,mean, fhwm/2.355)

def gaussian_fwhm(n, fwhm, binsize):
    std =  fwhm/(2.355*binsize)
    return scipy.signal.gaussian(n, std)/np.sqrt(2*np.pi*std**2)

binsize = 2
ka_energy = np.arange(6380, 6415,binsize)
kb_energy = np.arange(7020,7095,binsize)
fwhm = 8
ka_counts = 10e6
kb_counts = 0.1*ka_counts
excitation_fraction = 0.2


eipairs = {}
plt.figure()
for key in fnames.keys():
    data = np.loadtxt(fnames[key])
    if key.startswith("ka"):
        energy = ka_energy
    else:
        energy = kb_energy
    intensity = np.interp(energy, data[:,0], data[:,1])
    intensity -= intensity[0]
    intensity_convolve = np.convolve(intensity, gaussian_fwhm(len(intensity), fwhm, binsize),"same")
    intensity_convolve = np.abs(intensity_convolve)
    eipairs[key] = energy, intensity_convolve
    plt.plot(energy, intensity_convolve, label=key)

plt.legend()
plt.xlabel("energy (eV)")
plt.ylabel("counts per %g eV bins"%binsize)


#ka ratio plot
energy_hs, intensity_hs = eipairs["ka_hs"]
energy_ls, intensity_ls = eipairs["ka_ls"]
intensity_hs[energy_hs>=6405] = intensity_ls[energy_hs>=6405]
intensity_hs[energy_hs<=6390] = intensity_ls[energy_hs<=6390]

plt.figure()
plt.plot(energy_hs, intensity_hs/intensity_ls-1,label="high/low-1")
plt.plot(energy_hs, 0.1*intensity_ls/intensity_ls.mean(),label="low")
plt.xlabel("energy (eV)")
plt.ylabel("intensity/intensity ratio per %g eV bin"%binsize)
plt.title("fwhm=%g"%fwhm)
plt.grid("on")
plt.ylim([-0.2,0.5])


#ka sim plot
avg_hs = intensity_hs*ka_counts/intensity_hs.sum()
avg_ls = intensity_ls*ka_counts/intensity_ls.sum()
avg_unpumped = avg_hs
avg_pumped = (1-excitation_fraction)*avg_hs+excitation_fraction*avg_ls
counts_unpumped = np.array([np.random.poisson(n) for n in avg_unpumped])
counts_pumped = np.array([np.random.poisson(n) for n in avg_pumped])
plt.figure()
plt.subplot(211)
plt.plot(energy_hs, counts_unpumped, label="unpumped")
plt.plot(energy_hs, counts_pumped, label="pumped")
plt.ylabel("count/%g eV"%binsize)
plt.legend(loc="upper left")
plt.grid("on")
plt.title("k_alpha, fwhm=%g, counts=%g, excitation frac=%g"%(fwhm, 2*ka_counts, excitation_fraction))
plt.subplot(212)
plt.plot(energy_hs, (counts_pumped-counts_unpumped)/np.sqrt(1000+counts_pumped+counts_unpumped),".-")
plt.xlabel("energy (eV)")
plt.grid("on")
plt.ylabel("pumped-unpumped/sqrt(p+u)")


#kb ratio plot
energy_hs, intensity_hs = eipairs["kb_hs"]
energy_ls, intensity_ls = eipairs["kb_ls"]
intensity_hs[energy_hs>=7075] = intensity_ls[energy_hs>=7075]
intensity_hs[energy_hs<=7030] = intensity_ls[energy_hs<=7030]

plt.figure()
plt.plot(energy_hs, intensity_hs/intensity_ls-1,label="high/low-1")
plt.plot(energy_hs, 0.1*intensity_ls/intensity_ls.mean(),label="low")
plt.xlabel("energy (eV)")
plt.ylabel("intensity/intensity ratio per %g eV bin"%binsize)
plt.title("fwhm=%g"%fwhm)
plt.grid("on")
plt.ylim([-0.2,0.5])


#kb sim plot
avg_hs = intensity_hs*kb_counts/intensity_hs.sum()
avg_ls = intensity_ls*kb_counts/intensity_ls.sum()
avg_unpumped = avg_hs
avg_pumped = (1-excitation_fraction)*avg_hs+excitation_fraction*avg_ls
counts_unpumped = np.array([np.random.poisson(n) for n in avg_unpumped])
counts_pumped = np.array([np.random.poisson(n) for n in avg_pumped])
plt.figure()
plt.subplot(211)
plt.plot(energy_hs, counts_unpumped, label="unpumped")
plt.plot(energy_hs, counts_pumped, label="pumped")
plt.ylabel("count/%g eV"%binsize)
plt.legend(loc="upper left")
plt.grid("on")
plt.title("k_beta, fwhm=%g, counts=%g, excitation frac=%g"%(fwhm, 2*kb_counts, excitation_fraction))
plt.subplot(212)
plt.plot(energy_hs, (counts_pumped-counts_unpumped)/np.sqrt(1000+counts_pumped+counts_unpumped),".-")
plt.xlabel("energy (eV)")
plt.grid("on")
plt.ylabel("pumped-unpumped/sqrt(p+u)")