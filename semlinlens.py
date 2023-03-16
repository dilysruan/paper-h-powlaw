import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import hyp2f1
import emcee
import corner
from astropy import units as u
from astropy import constants as const



######################################################################
# general setup
######################################################################

# Pauli matrices needed for shear calculations
Pauli_s1 = np.array([[0,1],[1,0]])
Pauli_s3 = np.array([[1,0],[0,-1]])

# 2x2 identity matrix comes up in different places
I2 = np.eye(2)



######################################################################
# class to hold image data and perform lens modeling
######################################################################

class lens:

    ##################################################################
    # initialize with empty arrays
    ##################################################################
    zlens = -1.0
    zsrc = -1.0
    nsrc = 0
    nimg = 0
    xarr = []  # position
    xsig = []  # position uncertainties
    Sarr = []  # inverse covariance matrices
    farr = []  # flux
    fsig = []  # flux uncertainties
    tarr = []  # time delay
    tsig = []  # time delay uncertainties
    farr = []  # image fluxes
    # flux info
    fluxmode = 'none'
    f_signed = True
    # parameter modes
    Amode = 'b'
    hmode = 'h'
    # shear prior
    sigma_gamma = None
    shear_prior_fac = 1.0

    ##################################################################
    # read from a lensmodel data file and store key quantities;
    # NOTE: the cosmology should be defined with H0 = 100 km/s/Mpc,
    # as this code will apply the h scaling
    ##################################################################
    def __init__(self,filename,zlens,zsrc,cosmo,fluxmode='none',
        shearprior=None,Amode='b',hmode='h',verbose=True):

        # read the file
        f = open(filename,'rt')
        alllines = []
        for l in skip_comments(f):
            alllines.append(l)
        f.close()

        # initialize various variables
        self.nsrc = int(alllines[5][0])
        if self.nsrc!=1:
            print('ERROR: code cannot handle multiple sources')
            return
        self.nimg = 0
        self.xarr = []
        self.xsig = []
        self.farr = []
        self.fsig = []
        self.tarr = []
        self.tsig = []
        self.nsig = []

        # compute time scale tau
        self.zlens = zlens
        self.zsrc = zsrc
        Dol = cosmo.angular_diameter_distance(self.zlens)
        Dos = cosmo.angular_diameter_distance(self.zsrc)
        Dls = cosmo.angular_diameter_distance_z1z2(self.zlens,self.zsrc)
        tau = (1.0+self.zlens)/const.c*Dol*Dos/Dls
        # store in days, and include radian/arcsec conversion
        self.tau0 = tau.to(u.day).value*(np.pi/(3600.0*180.0))**2

        # galaxy position
        x0 = float(alllines[1][0])
        y0 = float(alllines[1][1])

        # loop over images
        iline = 6
        self.nimg = int(alllines[iline][0])
        iline += 1
        for iimg in range(self.nimg):
            x    = float(alllines[iline][0])
            y    = float(alllines[iline][1])
            f    = float(alllines[iline][2])
            xsig = float(alllines[iline][3])
            fsig = float(alllines[iline][4])
            tdel = float(alllines[iline][5])
            tsig = float(alllines[iline][6])
            # note that we work in coordinates centered on the galaxy
            self.xarr.append([x-x0,y-y0])
            self.xsig.append(xsig)
            self.farr.append(f)
            self.fsig.append(fsig)
            self.tarr.append(tdel)
            self.tsig.append(tsig)
            iline += 1
        self.xarr = np.array(self.xarr)
        self.xsig = np.array(self.xsig)
        self.farr = np.array(self.farr)
        self.fsig = np.array(self.fsig)
        self.tarr = np.array(self.tarr)
        self.tsig = np.array(self.tsig)

        # check whether fluxes are signed
        if np.any(self.farr<0.0):
            self.f_signed = True
        else:
            self.f_signed = False

        # sort in time delay order
        indx = np.argsort(self.tarr)
        self.xarr = self.xarr[indx]
        self.xsig = self.xsig[indx]
        self.farr = self.farr[indx]
        self.fsig = self.fsig[indx]
        self.tarr = self.tarr[indx]
        self.tsig = self.tsig[indx]

        # flux mode
        self.fluxmode = fluxmode

        # parameter modes
        self.Amode = Amode
        self.hmode = hmode

        # shear prior
        self.shear_prior(shearprior)

        # done
        if verbose: print('Read image data from file',filename)

    ######################################################################
    # specify (Gaussian) prior on the shear components
    ######################################################################

    def shear_prior(self,sigma_gamma):
        self.sigma_gamma = sigma_gamma

    ######################################################################
    # process parameters based on the specified modes
    ######################################################################

    def procparms(self,eta,bin,hin):
        if self.Amode=='b':
            bout = bin
        elif self.Amode=='Rein':
            bout = bin**(2.0-eta)
        else:
            print('ERROR: unknown value of Amode')
            return
        if self.hmode=='h':
            hout = hin
        elif self.hmode=='hhat':
            hout = hin*(2.0-eta)
        else:
            print('ERROR: unknown value of hmode')
            return
        return bout,hout

    ######################################################################
    # simplified circular analysis - solve for parameters by hand
    ######################################################################

    def lnP_circ(self,p):
        if self.nimg!=2:
            print('ERROR: circular analysis can handle only 2 images')
            return

        # extract parameters; note this b is dummy that gets overwritten below
        eta = p[0]
        btmp = 1.0
        htmp = p[1]
        b,h = self.procparms(eta,btmp,htmp)
        self.tau = self.tau0/h

        r = np.linalg.norm(self.xarr,axis=1)
        r1,r2 = r

        # solve for normalization and source position
        b = (r1+r2)/(r1**(eta-1.0)+r2**(eta-1.0))
        u = (r1*r2**(eta-1.0)-r2*r1**(eta-1.0))/(r1**(eta-1.0)+r2**(eta-1.0))

        # flux analysis
        phir_r = b*r**(eta-2.0)
        phirr = b*(eta-1.0)*r**(eta-2.0)
        mu = 1.0/((1.0-phir_r)*(1.0-phirr))
        if self.f_signed==False: mu = np.absolute(mu)
        f_a = np.sum(mu**2/self.fsig**2)
        f_b = np.sum(mu*self.farr/self.fsig**2)
        f_c = np.sum(self.farr**2/self.fsig**2)
        if self.fluxmode=='optimize':
            lnPflux = 0.5*(f_b**2/f_a) - 0.5*f_c
        elif self.fluxmode=='marginalize':
            lnPflux = 0.5*(f_b**2/f_a) - 0.5*f_c - 0.5*np.log(f_a)
        else:
            print('ERROR: fluxmode not recognized')

        # time delay analysis
        dt = 0.5*(r2**2-r1**2) + (r1+r2)*u - b/eta*(r2**eta-r1**eta)
        dt = self.tau*dt
        lnPtdel = -0.5*(dt-self.tarr[1])**2/self.tsig[1]**2

        return lnPflux+lnPtdel

    ######################################################################
    # calculate lnP using the full sum over images:
    # - fullsum0 : source position is optimized analytically
    # - fullsum1 : source position is treated as free parameters
    ######################################################################

    def lnP_fullsum0(self,pall):
        # extract parameters
        u = 0
        v = 0
        if self.ellip==True:
            gc,gs,ec,es,btmp,eta,htmp = pall
        else:
            gc,gs,btmp,eta,htmp = pall
            ec = 0
            es = 0
        # update b and h based on specified modes
        b,h = self.procparms(eta,btmp,htmp)
        # put them into appropriate lists
        plin = np.array([u, v, gc, gs, b])
        pnon = np.array([eta, ec, es])

        # make sure parameters stay in bounds
        if eta<0.1 or eta>1.9:
            return -1.0e8
        if h<0.0 or h>2.0:
            return -1.0e8

        # compute lensing quantities
        tdel,lens,Gamm = calc_lens(self.xarr,pnon,plin)

        # we need the product mu^T.S.mu for each image
        mSm = []
        for i in range(self.nimg):
            mu = np.linalg.inv(I2-Gamm[i])
            mSm.append(mu.T@mu/self.xsig[i]**2)
        mSm = np.array(mSm)

        # find best source position
        utmp = np.array([ mSm[i]@(self.xarr[i]-lens[i]) for i in range(self.nimg) ])
        rhs = np.sum(utmp,axis=0)
        lhs = np.sum(mSm,axis=0)
        ubest = np.linalg.solve(lhs,rhs)

        # compute lnP using this source position along with our params
        ptmp = np.concatenate((ubest,pall))
        lnP = self.lnP_fullsum1(ptmp)

        return lnP

    def lnP_fullsum1(self,pall):
        # extract parameters
        if self.ellip==True:
            u,v,gc,gs,ec,es,btmp,eta,htmp = pall
        else:
            u,v,gc,gs,btmp,eta,htmp = pall
            ec = 0
            es = 0
        # update b and h based on specified modes
        b,h = self.procparms(eta,btmp,htmp)
        # put them into appropriate lists
        plin = np.array([u, v, gc, gs, b])
        pnon = np.array([eta, ec, es])

        # make sure parameters stay in bounds
        if eta<0.1 or eta>1.9:
            return -1.0e8
        if h<0.0 or h>2.0:
            return -1.0e8

        # compute modified time delays
        self.tau = self.tau0/h
        r = np.linalg.norm(self.xarr,axis=1)
        xi = 0.5*self.tau*(r**2-r[0]**2) - self.tarr

        # compute lensing quantities
        tdel,lens,Gamm = calc_lens(self.xarr,pnon,plin)
        # compute magnifications
        muarr = []
        mu = []
        for i in range(self.nimg):
            tmp = np.linalg.inv(I2-Gamm[i])
            muarr.append(tmp)
            mu.append(np.linalg.det(tmp))
        muarr = np.array(muarr)
        mu = np.array(mu)
        # insert tau and recall that we need differential time delays
        tdel = self.tau*(tdel-tdel[0])

        chisq = 0.0

        # chisq from positions
        for i in range(self.nimg):
            mSm = muarr[i].T@muarr[i]/self.xsig[i]**2
            du = lens[i] - self.xarr[i]
            chisq += du@mSm@du

        # chisq from fluxes
        if self.f_signed==False: mu = np.absolute(mu)
        f_a = np.sum(mu**2/self.fsig**2)
        f_b = np.sum(mu*self.farr/self.fsig**2)
        f_c = np.sum(self.farr**2/self.fsig**2)
        if self.fluxmode=='optimize':
            chisq += -(f_b**2/f_a) + f_c
        elif self.fluxmode=='marginalize':
            chisq += -(f_b**2/f_a) + f_c + np.log(f_a)
        elif self.fluxmode=='none':
            chisq += 0.0
        else:
            print('ERROR: fluxmode not recognized')

        # chisq from time delays
        for i in range(1,self.nimg):
            dt = tdel[i] - xi[i]
            chisq += dt**2/self.tsig[i]**2

        return -0.5*chisq

    ######################################################################
    # calculate lnP from full parameter set, using matrix approach
    # - fullmatrix0 : mu factors from optimal values of linear parameters
    # - fullmatrix1 : mu factors from current values of linear parameters
    ######################################################################

    def lnP_fullmatrix0(self,pall):
        # extract parameters
        u = 0
        v = 0
        if self.ellip==True:
            gc,gs,ec,es,btmp,eta,htmp = pall
        else:
            gc,gs,btmp,eta,htmp = pall
            ec = 0
            es = 0
        # update b and h based on specified modes
        b,h = self.procparms(eta,btmp,htmp)
        # put them into appropriate lists
        plin = np.array([u, v, gc, gs, b])
        pnon = np.array([eta, ec, es])

        # make sure parameters stay in bounds
        if eta<0.1 or eta>1.9:
            return -1.0e8
        if h<0.0 or h>2.0:
            return -1.0e8

        # first optimize the nuisance parameters without mu factors
        self.setup_matrices(pnon,[])
        self.vbest = np.linalg.solve(self.Atot,self.wtot)

        # compute lensing quantities using vbest
        tdel,lens,Gamm = calc_lens(self.xarr,pnon,self.vbest)
        muinv = np.array([ I2-Gamm[i] for i in range(self.nimg) ])

        # recompute arrays, now with mu factors
        self.setup_matrices(pnon,muinv)

        # use the arrays to compute chisq
        chisq = plin@self.Atot@plin - 2.0*plin@self.wtot + self.Xtot

        return -0.5*chisq

    def lnP_fullmatrix1(self,pall):
        # extract parameters
        u = 0
        v = 0
        if self.ellip==True:
            gc,gs,ec,es,btmp,eta,htmp = pall
        else:
            gc,gs,btmp,eta,htmp = pall
            ec = 0
            es = 0
        # update b and h based on specified modes
        b,h = self.procparms(eta,btmp,htmp)
        # put them into appropriate lists
        plin = np.array([u, v, gc, gs, b])
        pnon = np.array([eta, ec, es])

        # make sure parameters stay in bounds
        if eta<0.1 or eta>1.9:
            return -1.0e8
        if h<0.0 or h>2.0:
            return -1.0e8

        # compute lensing quantities
        tdel,lens,Gamm = calc_lens(self.xarr,pnon,plin)
        muinv = np.array([ I2-Gamm[i] for i in range(self.nimg) ])

        # compute arrays
        self.setup_matrices(pnon,muinv)

        # use the arrays to compute chisq
        chisq = plin@self.Atot@plin - 2.0*plin@self.wtot + self.Xtot

        return -0.5*chisq

    ######################################################################
    # calculate lnP after Gaussian marginalization
    ######################################################################

    def lnP_marginal(self,pnon):

        # first optimize the nuisance parameters without mu factors
        self.setup_matrices(pnon,[])
        self.vbest = np.linalg.solve(self.Atot,self.wtot)

        # compute lensing quantities using vbest
        tdel,lens,Gamm = calc_lens(self.xarr,pnon,self.vbest)
        muinv = np.array([ I2-Gamm[i] for i in range(self.nimg) ])

        # re-optimize with the mu factors
        self.setup_matrices(pnon,muinv)
        self.vbest = np.linalg.solve(self.Atot,self.wtot)

        chisq = -self.vbest@self.wtot + self.Xtot
        s,logdet = np.linalg.slogdet(self.Atot)
        lnP = -0.5*chisq - 0.5*logdet

        # account for any normalization factor from shear prior
        lnP = lnP + np.log(self.shear_prior_fac)

        return lnP

    ######################################################################
    # compute arrays needed for full analysis
    ######################################################################

    def setup_matrices(self,pnon,muinv=[]):
        eta = pnon[0]
        h   = pnon[1]
        self.tau = self.tau0/h

        # construct Sarr, with or without mu factors as available
        self.Sarr = []
        if len(muinv)==0:
            for i in range(self.nimg):
                self.Sarr.append(I2/self.xsig[i]**2)
        else:
            for i in range(self.nimg):
                mu = np.linalg.inv(muinv[i])
                self.Sarr.append(mu.T@mu/self.xsig[i]**2)
        self.Sarr = np.array(self.Sarr)

        # Tarr is list of 1/sigma_t^2
        self.Tarr = 1.0/self.tsig**2

        # compute modified time delays
        self.tau = self.tau0/h
        r = np.linalg.norm(self.xarr,axis=1)
        xi = 0.5*self.tau*(r**2-r[0]**2) - self.tarr

        # lensing calculations
        dtdel,dlens,dGamm = calc_lens_deriv(self.xarr,[eta])
        # insert tau and recall that we need differential time delays
        for a in range(len(dtdel)):
            dtdel[a] = self.tau*(dtdel[a]-dtdel[a,0])

        # construct arrays; our indices are as follows:
        # - a,b = parameters
        # - i   = image
        # - x,y = spatial coordinate
        Apos = np.einsum('aix,ixy,biy',dlens    ,self.Sarr,dlens    )
        Adel = np.einsum('ai ,i  ,bi ',dtdel    ,self.Tarr,dtdel    )
        wpos = np.einsum('aix,ixy,iy ',dlens    ,self.Sarr,self.xarr)
        wdel = np.einsum('ai ,i  ,i  ',dtdel    ,self.Tarr,xi       )
        Xpos = np.einsum('ix ,ixy,iy ',self.xarr,self.Sarr,self.xarr)
        Xdel = np.einsum('i  ,i  ,i  ',xi       ,self.Tarr,xi       )

        # combine position and time delay constraints
        self.Atot = Apos + Adel
        self.wtot = wpos + wdel
        self.Xtot = Xpos + Xdel

        # impose any shear prior
        if self.sigma_gamma!=None:
            fac = 1.0/self.sigma_gamma**2
            self.Atot[3,3] = self.Atot[3,3] + fac
            self.Atot[4,4] = self.Atot[4,4] + fac
            # fac = sqrt(|S|) is the normalization factor
            self.shear_prior_fac = fac

    ######################################################################
    # wrapper to compute probability for an array of parameters
    ######################################################################

    def prob(self,parr=[],erange=[],hrange=[],mode='',return_p=False):

        # if parr is not specified, build grid from erange and hrange
        if len(parr)==0:
            etmp = np.linspace(erange[0],erange[1],erange[2])
            htmp = np.linspace(hrange[0],hrange[1],hrange[2])
            parr = np.moveaxis(np.array(np.meshgrid(etmp,htmp)),0,-1)

        # store shape of parr so we can apply it to the answer array
        parr = np.array(parr)
        pshape = parr.shape

        # loop through flattened list of parameters
        ptmp = parr.reshape((-1,pshape[-1]))
        lnParr = []
        for p in ptmp:
            if mode=='circ':
                lnPtmp = self.lnP_circ(p)
            elif mode=='marginal':
                lnPtmp = self.lnP_marginal(p)
            else:
                print('ERROR: unknown mode in prob function')
                return
            lnParr.append(lnPtmp)
        lnParr = np.array(lnParr)

        # impose the desired shape
        prob_arr = np.exp(lnParr.reshape(pshape[:-1]))

        if return_p:
            return prob_arr,parr
        else:
            return prob_arr

    ######################################################################
    # wrapper for optimize
    ######################################################################

    def optimize_func(self,p,mode):
        if mode=='fullsum0':
            lnlike = self.lnP_fullsum0(p)
        elif mode=='fullsum1':
            lnlike = self.lnP_fullsum1(p)
        elif mode=='fullmatrix0':
            lnlike = self.lnP_fullmatrix0(p)
        elif mode=='fullmatrix1':
            lnlike = self.lnP_fullmatrix1(p)
        elif mode=='marginal':
            lnlike = self.lnP_marginal(p)
        elif mode=='circ':
            lnlike = self.lnP_circ(p)
        else:
            print('ERROR: MCMC mode not recognized')
            lnlike = -2.0e12
        return -2.0*lnlike

    def optimize(self,pstart,mode='',ellip=False):
        self.ellip = ellip
        ans = minimize(self.optimize_func,pstart,args=(mode,))
        return ans.x,ans.fun,ans.success
    
    def opt_MCin(self, nopt, basename,outname,zlens, zsrc, ellip=False,eta_range=[0.5,1.5], mode='fullsum1', Amode='b', hmode='h', fluxmode='none'):
        
        # compute the average distance of images from the origin; this is useful for guessing initial conditions
        ravg = np.mean(np.linalg.norm(self.xarr,axis=1))
        print(f'<r> = {ravg:.3f}')

        # optimize from a bunch of random starting points
        print('Optimizing')
        chibest = 1.0e8
        pbest = []
        for iran in range(nopt):
            # check shear and ellipticity in Gaussian distributions
            gc = np.random.normal(scale=0.05)
            gs = np.random.normal(scale=0.05)
            ec = np.random.normal(scale=0.1)
            es = np.random.normal(scale=0.1)
            # check eta in the specified range
            eta = np.random.uniform(low=eta_range[0],high=eta_range[1])
            # guess b from eta and ravg
            b = ravg**(2.0-eta)
            # always guess h = 0.7
            h = 0.7
            if ellip==False and mode=='fullsum1':
                p0 = [0,0,gc,gs,b,eta,h]
            elif ellip==True and mode=='fullsum1':
                p0 = [0,0,gc,gs,ec,es,b,eta,h]
            elif ellip==False and mode=='fullsum0':
                p0 = [gc,gs,b,eta,h]
            elif ellip==True and mode=='fullsum0':
                p0 = [gc,gs,ec,es,b,eta,h]
                
            # optimize from this guess
            opt = self.optimize(p0,mode=mode,ellip=ellip)
            # if this is the best we have found so far, store it
            if opt[1]<chibest:
                pbest = opt[0]
                chibest = opt[1]
                print('best so far:',chibest,pbest)

        print('best optimization:')
        print(pbest)
        print('chisq',chibest)
        optimized = [float(chibest)]
        for n in range(len(pbest)):
            optimized.append(pbest[n])
        np.savetxt(outname+'-best.txt', optimized)
        return pbest

    ######################################################################
    # wrapper for emcee
    ######################################################################

    def runMC(self,pstart=[],mode='',nwalk=30,nburn=8000,nmain=8000,
        plotchains=False,ellip=False,outname='',savecorner=True):

        print('Running MC')
        self.ellip = ellip

        if self.Amode=='b':
            Aparam = r'$b$'
        elif self.Amode=='Rein':
            Aparam = r'$R_{\rm ein}$'

        if self.hmode=='h':
            hparam = r'$h$'
        elif self.hmode=='hhat':
            hparam = r'${\hat h}$'

        plab_src = [r'$u$', r'$v$']
        pini_src = [0, 0]
        plab_shr = [r'$\gamma_c$', r'$\gamma_s$']
        pini_shr = [0, 0]
        if self.ellip==True:
            plab_gal = [r'$e_c$', r'$e_s$', Aparam]
            pini_gal = [0, 0, 1]
        else:
            plab_gal = [Aparam]
            pini_gal = [1]
        plab_non = [r'$\eta$', hparam]
        pini_non = [1.0, 0.7]
        if mode=='fullsum0':
            plabel = plab_shr + plab_gal + plab_non
            pinit  = pini_shr + pini_gal + pini_non
            lnlike = self.lnP_fullsum0
        elif mode=='fullsum1':
            plabel = plab_src + plab_shr + plab_gal + plab_non
            pinit  = pini_src + pini_shr + pini_gal + pini_non
            lnlike = self.lnP_fullsum1
        elif mode=='fullmatrix0':
            plabel = plab_src + plab_shr + plab_gal + plab_non
            pinit  = pini_src + pini_shr + pini_gal + pini_non
            lnlike = self.lnP_fullmatrix0
        elif mode=='fullmatrix1':
            plabel = plab_src + plab_shr + plab_gal + plab_non
            pinit  = pini_src + pini_shr + pini_gal + pini_non
            lnlike = self.lnP_fullmatrix1
        elif mode=='marginal':
            plabel = plab_non
            pinit  = pini_non
            lnlike = self.lnP_marginal
        elif mode=='circ':
            plabel = plab_non
            pinit  = pini_non
            lnlike = self.lnP_circ
        else:
            print('ERROR: MCMC mode not recognized')
            return

        # use pstart if supplied, otherwise use pinit
        if len(pstart)==0: pstart = pinit

        # starting points are perturbed from pstart
        ndim  = len(pstart)
        p0 = pstart + 1.0e-3*np.random.normal(size=(nwalk,ndim))

        # set up the sampler
        sampler = emcee.EnsembleSampler(nwalk,ndim,lnlike)

        # burn-in run
        print('emcee burn-in run')
        pos,prob,state = sampler.run_mcmc(p0,nburn)

        # reset the sampler
        sampler.reset()

        # main run
        print('emcee main run')
        res = sampler.run_mcmc(pos,nmain)

        # if desired, plot the chains as a way to check for convergence
        if plotchains:
            f,ax = plt.subplots(ndim,1,figsize=(10,8),sharex=True)
            for idim in range(ndim):
                for iwalk in range(nwalk):
                    ax[idim].plot(sampler.chain[iwalk,:,idim])
                ax[idim].set_ylabel(plabel[idim])
            ax[-1].set_xlabel('step')
            f.show()

        # corner plot
        samples = sampler.chain.reshape((-1,ndim))
        f = corner.corner(samples,show_titles=True,title_fmt='.3f',labels=plabel)
        
        if len(outname)>0 or savecorner==True:
            f.savefig(outname+'.pdf',bbox_inches='tight')
        else:
            f.show()
        
        # check optimizing from one of the samples
        np.savetxt(outname+'.txt', samples)
        print('final optimization:')
        opt = self.optimize(samples[0],mode=mode,ellip=ellip)
        print(opt[1],opt[0])

        return samples

    ######################################################################
    # utility
    ######################################################################

    def Sfunc(self,v1,v2):
        ans = 0
        for i in range(self.nimg):
            ans = ans + v1[i]@self.Sarr[i]@v2[i]
        return ans

    def Tfunc(self,v1,v2):
        ans = 0
        for i in range(1,self.nimg):
            ans = ans + v1[i]*v2[i]/self.tsig[i]**2
        return ans




######################################################################
# lensing calculations - circular or elliptical power law,
# with external shear
#
# parameters: u,v,gc,gs,ec,es,b,eta,h
# - in circular models, ec=es=0; these may be omitted in functions
#   above, but included as zeros here
#
# linear parameters = [u, v, gc, gs, b]
# non-linear parameters = [eta, ec, es]
######################################################################

def calc_lens_deriv(xarr,pnon):
    # non-linear model parameter(s)
    eta = pnon[0]
    ec = pnon[1]
    es = pnon[2]
    ellip = np.sqrt(ec**2+es**2)

    # store shape of xarr so we can apply it to the results arrays
    xarr = np.array(xarr)
    xshape = xarr.shape[:-1]

    # flatten for ease of use here
    x = xarr.reshape((-1,2))
    nimg = len(x)

    # derivs wrt source position
    dtdel_du = x[:,0]
    dtdel_dv = x[:,1]
    dlens_du = np.array([ [1,0] for iimg in range(nimg) ])
    dlens_dv = np.array([ [0,1] for iimg in range(nimg) ])
    dGamm_du = np.zeros((nimg,2,2))
    dGamm_dv = np.zeros((nimg,2,2))

    if ellip==0:

        # circular power law galaxy
        r      = np.linalg.norm(x,axis=1)
        cost   = x[:,0]/r
        sint   = x[:,1]/r
        phi    = r**eta/eta
        phir_r = r**(eta-2.0)
        phirr  = phir_r*(eta-1.0)
        phixx  = phir_r*sint*sint + phirr*cost*cost
        phiyy  = phir_r*cost*cost + phirr*sint*sint
        phixy  = (phirr-phir_r)*sint*cost

        # derivs wrt b
        dtdel_db = phi
        dlens_db = np.array([ phir_r[i]*x[i] for i in range(nimg) ])
        dGamm_db = np.moveaxis(np.array([[phixx,phixy],[phixy,phiyy]]),-1,0)

    else:

        # elliptical power law model
        # kappa = (1/2) * eta * b/R^(2-eta)
        # where R = sqrt(q^2*x^2+y^2) is elliptical radius
        # Analysis by Tessore & Metcalf:
        # https://ui.adsabs.harvard.edu/abs/2015A%26A...580A..79T/abstract
        # https://ui.adsabs.harvard.edu/abs/2016A%26A...593C...2T/abstract (erratum)

        # index used by Tessore & Metcalf
        t = 2.0-eta
        # note that I use bt = b^t

        # process ellipticity and orientation
        e = np.sqrt(ec**2+es**2)
        q = 1.0-e
        te = 0.5*np.arctan2(es,ec)
        rot = np.array([[np.cos(te),-np.sin(te)],[np.sin(te),np.cos(te)]])

        # coordinates centered on and aligned with ellipse
        dx = x@rot

        # complex coordinate
        z = dx[:,0] + 1j*dx[:,1]
        z_conj = np.conj(z)

        # elliptical radius and angle defined by Tessore & Metcalf
        R = np.sqrt((q*dx[:,0])**2+dx[:,1]**2)
        phi = np.arctan2(dx[:,1],q*dx[:,0])

        e_iphi = np.cos(phi) + 1j*np.sin(phi)
        e_iphi_conj = np.conj(e_iphi)
        e_i2phi = np.cos(2*phi) + 1j*np.sin(2*phi)
        e_i2phi_conj = np.conj(e_i2phi)

        # deflection (complex)
        alpha_R = 2/((1+q)*R**(t-1))
        alpha_ang = e_iphi*hyp2f1(1,0.5*t,2-0.5*t,-(1-q)/(1+q)*e_i2phi)
        alpha_comp = alpha_R*alpha_ang
        alpha_conj = np.conj(alpha_comp)

        # potential
        phi = (z*alpha_conj+z_conj*alpha_comp)/(2*(2-t))

        # convergence and shear (complex)
        kappa = (2-t)/(2*R**t)
        gamma_comp = -kappa*z/z_conj + (1-t)*alpha_comp/z_conj

        # revert to vector/matrix notation; here the last index is position
        atmp = np.array([alpha_comp.real,alpha_comp.imag])
        Gtmp = np.array([[kappa+gamma_comp.real,gamma_comp.imag],[gamma_comp.imag,kappa-gamma_comp.real]])

        # handle rotation and reorder to get list of vectors/matrices;
        # this gives derivs wrt b
        dtdel_db = phi.real
        dlens_db = np.einsum('ij,ja',rot,atmp)
        dGamm_db = np.einsum('ij,jka,lk',rot,Gtmp,rot)

    # derivs wrt shear
    # products of positions with Pauli matrices
    s1x = x@Pauli_s1
    s3x = x@Pauli_s3
    dtdel_dgc = np.array([ 0.5*x[i]@s3x[i] for i in range(nimg) ])
    dtdel_dgs = np.array([ 0.5*x[i]@s1x[i] for i in range(nimg) ])
    dlens_dgc = s3x
    dlens_dgs = s1x
    dGamm_dgc = np.array([ Pauli_s3 for i in range(nimg) ])
    dGamm_dgs = np.array([ Pauli_s1 for i in range(nimg) ])

    # combine the pieces
    dtdel = np.array([dtdel_du,dtdel_dv,dtdel_dgc,dtdel_dgs,dtdel_db])
    dlens = np.array([dlens_du,dlens_dv,dlens_dgc,dlens_dgs,dlens_db])
    dGamm = np.array([dGamm_du,dGamm_dv,dGamm_dgc,dGamm_dgs,dGamm_db])

    # reshape so the spatial parts match xshape
    tdel_shape = np.concatenate(([5],xshape))
    lens_shape = np.concatenate(([5],xshape,[2]))
    Gamm_shape = np.concatenate(([5],xshape,[2,2]))
    dtdel = np.reshape(dtdel,tdel_shape)
    dlens = np.reshape(dlens,lens_shape)
    dGamm = np.reshape(dGamm,Gamm_shape)

    return dtdel,dlens,dGamm

def calc_lens(xarr,pnon,plin):
    plin = np.array(plin)
    dtdel,dlens,dGamm = calc_lens_deriv(xarr,pnon)
    tdel = np.tensordot(plin,dtdel,(0,0))
    lens = np.tensordot(plin,dlens,(0,0))
    Gamm = np.tensordot(plin,dGamm,(0,0))
    return tdel,lens,Gamm



######################################################################
# utility
######################################################################

######################################################################
# read a file and skip lines that are blank or start with comment '#';
# this is modified from
# https://stackoverflow.com/questions/1706198/python-how-to-ignore-comment-lines-when-reading-in-a-file
######################################################################

def skip_comments(file):
    for line in file:
        if not line.strip().startswith('#'):
            if len(line.split())>0:
                yield line.split()
