# fishlearning

### addNNOri.py
Creates 'nori.btf' trace. Each line is a list of relative orientation of all neighbors for the given fish. No threshold on distance, so the 'nn' part is kind of false. Requires

- id.btf
- xpos.btf
- ypos.btf
- timage.btf
- timestamp.btf

and write permission to the given output directory.

### computeRBFOri.py
Creates 'rbforivec-*.btf' trace. Each line is the weighted average relative orientation of all fish. Threshold on distance is soft, parameterized by sigma. Requires 'nori.btf' and 'nvec.btf' (NOT 'nnvec.btf').

### computeRBFSep.py
Creates 'rbfsepvec-*.btf' trace. Each line is the weighted average relative position of all fish. Threshold on distance is soft, parameterized by sigma. Requires 'nvec.btf' (NOT 'nnvec.btf').

### dispObst2DVel.py
Simple script for displaying 'dvel.btf' as a function of 'wallvec.btf' in 2D and 3D using matplotlib.

### lr_reynolds.py
Old script, to be removed in future commits, mostly supplanted by 'sigma_opt.py'.

Would use 'rbf*vec-*.btf' traces to run linear regression and compute the residuals as a rough measure of performance.

### sigma_opt.py
Main script for computing linear regression parameters and optimizing values for sigma for each feature (separation, orientation, coheasion). 

With the change to scipy.optimize.basinhopping, currently does not respect evaluation limits, and takes a very long time (hour(s) per outer evaluation step) to run optimization. Linear regression can run quicker (10s of minutes).
