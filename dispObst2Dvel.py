import numpy, scipy, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sigma_opt

def main():
	dvel = numpy.array(map(lambda l: map(float,l.split()), open('dvel.btf').readlines()))
	wallvec = numpy.array(map(lambda l: map(float,l.split()), open('wallvec.btf').readlines()))
	tmpwallv = wallvec[:1000,:]
	tmpdvel = dvel[:tmpwallv.shape[0],:]
	weights = sigma_opt.weights(tmpwallv[:,0],tmpwallv[:,1],1.0)
	# walldist.btf got ctrl-c'd before it finished... I think
	# walldist = numpy.array(map(lambda l: map(float,l.split()),open('walldist.btf').readlines()))
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter(weights*tmpwallv[:,0],weights*tmpwallv[:,1],tmpdvel[:,0])
	fig.show()
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter(weights*tmpwallv[:,0],weights*tmpwallv[:,1],tmpdvel[:,1])
	fig.show()
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter(weights*tmpwallv[:,0],weights*tmpwallv[:,1],tmpdvel[:,2])
	fig = plt.figure()
	ax = fig.add_subplot()
	plt.scatter(weights*tmpwallv[:,0],weights*tmpwallv[:,1])
	fig.show()
	
if __name__ == '__main__':
	main()
	raw_input("Hit enter to close")
