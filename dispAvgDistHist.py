import numpy, scipy, matplotlib.pyplot as plt, sys
import sigma_opt
from scipy.spatial import cKDTree

def main(nvec_fname, nori_fname, obs_fname, dvel_fname, dbool_fname):
	print "Loading data"
	sigma_opt.load_data(nvec_fname, nori_fname, obs_fname, dvel_fname)
	print "Filtering data"
	sigma_opt.filter(dbool_fname)
	print "Weighting data"
	data = numpy.array(sigma_opt.weightedData([0.1,1.5,1.5,1.5]))
	print "Filling KDT"
	knn = cKDTree(features)
	print "Computing avg NN distance"
	dists = numpy.array([knn.query(row,k=10)[0].mean() for row in data])
	print "Plotting"
	plt.hist(dists,bins=50,histtype='stepfilled')
	plt.show()

if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
	raw_input("Hit enter to close")