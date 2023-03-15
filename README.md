# paper-h-powlaw

Code that compliments the paper: (in prep). I don't write code in the most optimized way, but this is meant to demonstrate how we did our Markov Chain Monte Carlo analysis.

The mock lens systems are generated with <a href="https://github.com/chuckkeeton/pygravlens">pygravlens</a>, a python version of <a>lensmodel</a>, written by Chuck Keeton. You can install this through the linked GitHub page.

To run this code, you will also need some common python packages:
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

Mock lenses can just be generated through pygravlens alone. The func_get_rel.py code simply outputs the relative time delays and annulus lengths for multiple lens systems in the .dat file. The MCMC analysis must be done using the likelihood functions defined in the semlinlens.py file. The func_get_h_eta.py code takes the MCMC output and finds the median and errors (within the 68% confidence interval) for h and the power law slope.
