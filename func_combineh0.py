import numpy as np
import matplotlib.pyplot as plt
import corner
import pandas as pd
import matplotlib.cm as cm
import random

from copy import deepcopy

from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from astropy.cosmology import FlatLambdaCDM
import statistics as stats
import scipy

import func_get_h_eta as h_eta
import func_get_rel_CK as rel

plt.rcParams.update({'font.size': 17, 'text.usetex': False})
dpi_set=400
cosmo = FlatLambdaCDM(H0=100,Om0=0.315)

def combineh0(folder, root, Nlens, color, choose = 'ellip', kind = 'threshlow', kind2 = '80', title = 'EPL+XS img,td noise', savelabel = 'EPLXSv3imgtd', Re=1.0):
    '''
    function combines the h pdf for each lens system for an overall 'most likely' pdf. Selects the data based on annulus length r12 between images with the first and second time delays.
    
    folder:   str for directory with MCMC chains with .txt
    root:     str for file name, enter "EPLXS_" for files like "EPLXS_1.txt"
    choose:   str for 'ellip' or 'circ'
    kind:     str for what kind of cut to make, 'all', 'threshlow', 'random20', 'random25', '1sigma' (above & below)
    kind2:    str for what to cut at & below, so 'med' lower 50, '75' lower 75, '80' lower 80)
    title:    str for title on h pdf plot
    savelabel:str for file root when saving plot
    Re:       float for Einstein radius in arcsec (default 1.0")
    
    function will output .txt file with h results, .txt file with threshold cuts, .npy with all h pdfs, .npy with combined pdf bins, .npy with combined pdf, and .npy with flags used
    '''
    folders = [folder]
    roots = [root]
    
    start = 0
    end = 1

    mocks=[]
    for n in range(1,Nlens+1):
        mocks.append(str(n))

    ### GATHER DATA FROM MCMC RUNS FOR EACH LENS SYSTEM
    combine_dat = []
    for nn in range(start,end):#
        folder = folders[nn]
        root = roots[nn]
        for n in range(0,len(mocks)):
            combine_dat.append(folder+root+mocks[n]+'.dat')

    Re_arr = [Re]*Nlens      
      
    imgpos, drarr, taumat = rel.get_info(combine_dat)
    ann14 = drarr[:,3] / Re_arr
    ann13 = drarr[:,2] / Re_arr
    ann12 = drarr[:,1] / Re_arr
    tau14 = taumat[:,0,3]
    tau13 = taumat[:,0,2]
    tau12 = taumat[:,0,1]

    ### FLAG BY WHATEVER QUANTITY
    x_flag = ann12
    flags = []

    ann_20 = np.sort(ann12)[79]
    ann_25 = np.sort(ann12)[74]

    med = stats.median(x_flag)
    error = (stats.stdev(x_flag))
    mean = stats.mean(x_flag)
    n_sigma = 1
    uppercut = med + n_sigma*error
    lowercut = med - n_sigma*error

    if kind2 == '80':
        thresh = ann_20
    elif kind2 == '75':
        thresh = ann_25
    elif kind2 == 'med':
        thresh = med

    print('median is: '+str(med))
    print('std dev is:'+str(error))
    print('mean is:'+str(mean))

    if kind == 'threshlow':
        for n in range(0,len(x_flag)):
            if x_flag[n] <= thresh:
                flags.append(False)
            else:
                flags.append(True)
        print('cutoff at below: '+str(round(thresh,3)))

    elif kind == '1sigma':
        for n in range(0,len(x_flag)):
            if x_flag[n] <= lowercut or x_flag[n] >= uppercut:
                flags.append(False)
            else:
                flags.append(True)
        print('lowercut, uppercut')
        print(lowercut, uppercut)

    elif kind == 'all':
        flags = [True]*Nlens

    elif kind == '1sigmalow':
        for n in range(0,len(x_flag)):
            if annulus_length[n] <= lowercut:
                flags.append(False)
            else:
                flags.append(True)

    elif kind == 'random25':
        Nsize = 25
        ind = ind = np.random.choice(range(Nlens), Nsize, replace=False)
        flags = [False]*Nlens
        for n in range(0,len(ind)):
            flags[ind[n]] = True
        np.savetxt(folder+savelabel+'random25.txt', ind)
        #print(ind)
        
    elif kind == 'random20':
        Nsize = 20
        ind = ind = np.random.choice(range(Nlens), Nsize, replace=False)
        flags = [False]*Nlens
        for n in range(0,len(ind)):
            flags[ind[n]] = True
        np.savetxt(folder+savelabel+'random25.txt', ind)

    print(flags)
    print(sum(flags))
    
    print(ind)

    ### READ IN EACH H PDF
    combine_dat = []
    combine_txt = []
    combine_best = []
    h_prob = []
    eta_prob = []
    heta_ind = 0
    for nn in range(start,end):#
        folder = folders[nn]
        root = roots[nn]
        for n in range(0,len(mocks)):
            combine_dat.append(folder+root+mocks[n]+'.dat')
            combine_txt.append(folder+root+mocks[n]+'.txt')
            combine_best.append(folder+root+mocks[n]+'-best.txt')
            if flags[n] == True:
                data = np.loadtxt(combine_txt[n])
                data = np.transpose(data)
                if choose == 'circ':
                    h_prob.append(data[6])
                    eta_prob.append(data[5])
                elif choose == 'ellip':
                    h_prob.append(data[8])
                    eta_prob.append(data[7])
                heta_ind += 1

    bin_num = 5000
    list_bins = np.linspace(0.5, 1.5, bin_num)
    Pall = np.ones(len(list_bins))
    
    # COMBINE PROBABILITIES
    kde_pdf = []
    number = 0
    for n in range(0,len(h_prob)):
        if flags[n] == True:
            kde_func = gaussian_kde(h_prob[number]) # smooth & interpolate the pdf for each
            kde_pdf.append(kde_func(list_bins))
            Pall *= kde_pdf[number] # multiply to combine the pdfs
            number += 1 

    # normalize distribution
    norm = np.sum(Pall)*(list_bins[1]-list_bins[0])
    Pall /= norm

    print(flags)
    print(number)

    np.save(folder+savelabel+'_'+kind+kind2+'_flags.npy', flags, allow_pickle = True) 
    np.savetxt(folder+savelabel+'_'+kind+kind2+'_cuts.txt', [med, ann_25, ann_20])
        
    np.save(folder+savelabel+'_'+kind+kind2+'_listbins.npy', list_bins, allow_pickle = True)
    np.save(folder+savelabel+'_'+kind+kind2+'_Pall.npy', Pall, allow_pickle = True)
    np.save(folder+savelabel+'_'+kind+kind2+'hprob.npy', h_prob, allow_pickle=True)
    print(root, kind, folder)

    list_bins = np.load(folder+savelabel+'_'+kind+kind2+'_listbins.npy', allow_pickle = True) 
    Pall = np.load(folder+savelabel+'_'+kind+kind2+'_Pall.npy', allow_pickle = True)
    hprob = np.load(folder+savelabel+'_'+kind+kind2+'hprob.npy', allow_pickle = True)

    plt.rcParams.update({'font.size': 14, 'text.usetex': False})
    plt.rc('axes', labelsize=14)
        
    plt.figure(dpi=300)
    bin_num_small = 50
    number = 0
    for n in range(0,len(mocks)):
        if flags[n] == True:
            num, bins = np.histogram(h_prob[number], bins=bin_num_small, density=True)
            plt.plot(bins[:-1], num, color=color, alpha=0.6)
            number+=1

    plt.plot(list_bins, Pall, color='black', label='combined', linewidth=3)
    plt.legend()

    plt.xlabel('h')
    plt.ylabel('prob density function')

    # cdf and interpolation to get the med and errors within 68% confidence interval
    cdf = (np.cumsum(Pall)*(list_bins[1]-list_bins[0]))
    cdf_interp = scipy.interpolate.interp1d(cdf, list_bins)
    x_50 = cdf_interp(0.5)
    x_16 = cdf_interp(0.16)
    x_84 = cdf_interp(0.84)
    lower_err = np.abs(x_50 - x_16)
    upper_err = np.abs(x_50 - x_84)

    bias = np.abs(0.7-x_50)
    rel_bias = round((bias/0.7*Nlens),2)

    print(x_50, x_16, x_84)
    print(lower_err, upper_err)
    np.savetxt(folder+savelabel+'_'+kind+kind2+'_hresult.txt', [x_16, x_50, x_84, rel_bias])

    index_maxpdf = np.argsort(Pall)[-1]

    length = 10
    color = 'black'
    plt.plot([x_16]*length, np.linspace(np.min(Pall), np.max(Pall)+10,10), linestyle='--', color=color)
    plt.plot([x_84]*length, np.linspace(np.min(Pall), np.max(Pall)+10,10), linestyle='--', color=color)
    plt.plot([x_50]*length, np.linspace(np.min(Pall), np.max(Pall)+10,10), linestyle='--', color=color)
    plt.xlim([0.5,1.15])
    plt.ylim([0,11])
    plt.title(title+', rel bias: '+f'{rel_bias:.2f}'+'%, h='+f'{x_50:.3f}'+' $\pm$'+f'{upper_err:.3f}', fontsize=14)

    plt.savefig(folder+'plot-'+savelabel+'_'+kind+kind2+'.pdf', bbox_inches='tight')