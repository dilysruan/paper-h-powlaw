# Statistical Bias in the Hubble Constant and Mass Power Law Slope for Mock Strong Lenses

Code that compliments the paper: Ruan & Keeton 2023, meant to demonstrate how we generated mock lenses and implemented the lens modelling.

The mock lens systems are generated with <a href="https://github.com/chuckkeeton/pygravlens">pygravlens</a>, a python version of <a href="https://www.physics.rutgers.edu/~keeton/gravlens/2012WS/">lensmodel</a>, written by Chuck Keeton. You can install pygravlens through the linked GitHub page.

To run this code, you will need these common python packages:
<ul>
  <li>numpy</li>
  <li>matplotlib</li>
  <li>scipy</li>
  <li>pandas</li>
  <li>astropy</li>
  <li>copy</li>
  <li>random</li>
  <li>statistics</li>
  <li> <a href="https://corner.readthedocs.io/en/latest/install/">corner</a> </li>
  <li> <a href="https://emcee.readthedocs.io/en/stable/">emcee</li>
</ul>

Mock lenses can just be generated through pygravlens alone (generate_mock_lenses.ipynb). The lens modelling (running MCMC, combining likelihoods) is done in one notebook (lens_modelling.ipynb).

The func_get_rel_CK.py code simply outputs the relative time delays and annulus lengths for multiple lens systems from different .dat files. The MCMC analysis must be done using the likelihood functions defined in the semlinlens_v3.py file. The func_get_h_eta.py code takes the MCMC output and finds the median and errors (within the 68% confidence interval) for h and the power law slope. The func_combineh0.py code will combine the MCMC chains and use gaussian kernel density estimation for each h posterior, and then multiply these (given whatever selection of the data you request) for an overall h posterior.
