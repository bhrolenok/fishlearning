import numpy, scipy, matplotlib.pyplot as plt, sys, glob
import sigma_opt
import btfutil
from scipy.spatial import cKDTree

# def main2(rbfsep_fname, rbfori_fname, rbfcoh_fname, rbfwall_fname):
# 	rbfsep = [line.split() for line in open(rbfsep_fname).readlines()]
# 	rbfori = [line.split() for line in open(rbfori_fname).readlines()]
# 	rbfcoh = [line.split() for line in open(rbfcoh_fname).readlines()]
# 	rbfwall = [line.split() for line in open(rbfwall_fname).readlines()]
# 	return numpy.column_stack([rbfsep,rbfori,rbfcoh,rbfwall])

# def main(nvec_fname, nori_fname, obs_fname, dvel_fname, dbool_fname):
# def main(rbfsep_fname, rbfori_fname, rbfcoh_fname, rbfwall_fname):
def main(btfdata):
	# print "DEBUG: MAXLINES=200"
	# sigma_opt.DEBUG_MAXLINES = 200
	data = None
	if btfdata.has_columns(['rbfsepvec','rbforivec','rbfcohvec','rbfwallvec']):
		rawdata = numpy.column_stack([\
			[line.split() for line in btfdata['rbfsepvec']],\
			[line.split() for line in btfdata['rbforivec']],\
			[line.split() for line in btfdata['rbfcohvec']],\
			[line.split() for line in btfdata['rbfwallvec']]])
		data = rawdata[[line.capitalize() == 'True' for line in btfdata['dbool']]]
	else:
		sigma_opt.load_data(\
			btfdata.column_filenames['nvec'],\
			btfdata.column_filenames['nori'],\
			btfdata.column_filenames['wallvec'],\
			btfdata.column_filenames['dvel'])
		sigma_opt.filter_dbool(btfdata.column_filenames['dbool'])
		print "Weighting data"
		data = numpy.column_stack(sigma_opt.weightedData([0.1,1.5,1.5,1.5]))
	print "Filling KDT"
	knn = cKDTree(data)
	print "Computing avg NN distance"
	# So aparantly query returns the neighbors in sorted order
	# so we should be able to get the distances excluding the point
	# itself by ignoring the first column
	# queries = numpy.array([knn.query(row,k=10)[0][1:] for row in data])
	dists = numpy.array([knn.query(row,k=10)[0][1:].mean() for row in data])
	print "Plotting"
	plt.hist(dists,bins=50,histtype='stepfilled')
	plt.show()
	print "Mean", dists.mean()
	print "Std. dev", dists.std()

if __name__ == '__main__':
	# main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
	# main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
	data = btfutil.BTF()
	data.import_from_dir(sys.argv[1])
	main(data)
	# older version of matplotlib didn't hold the graph unless I added this line
	# raw_input("Hit enter to close")