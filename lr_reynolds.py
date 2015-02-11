#lr_reynolds.py
import numpy, numpy.linalg, cPickle

def computeLRcombined(sep_name, align_name, cohes_name, b_name):
	v_sep = numpy.array([map(float,l.split()) for l in open(sep_name).readlines()])
	v_align = numpy.array([map(float,l.split()) for l in open(align_name).readlines()])
	v_cohes = numpy.array([map(float,l.split()) for l in open(cohes_name).readlines()])
	v_obst = numpy.array([map(float,l.split()) for l in open('wallvec.btf').readlines()])
	b = numpy.array([map(float,l.split()) for l in open(b_name).readlines()])
	return numpy.linalg.lstsq(numpy.column_stack([v_sep,v_align,v_cohes,v_obst,numpy.ones(v_sep.shape[0:1])]), b)

def computeLRindependent(sep_name, align_name, cohes_name, b_name):
	v_sep = numpy.array([map(float,l.split()) for l in open(sep_name).readlines()])
	v_align = numpy.array([map(float,l.split()) for l in open(align_name).readlines()])
	v_cohes = numpy.array([map(float,l.split()) for l in open(cohes_name).readlines()])
	v_obst = numpy.array([map(float,l.split()) for l in open('wallvec.btf').readlines()])
	b = numpy.array([map(float,l.split()) for l in open(b_name).readlines()])
	x_res = numpy.linalg.lstsq(numpy.column_stack([v_sep, v_align, v_cohes,v_obst, numpy.ones(v_sep.shape[0:1])]), b[:,0])
	y_res = numpy.linalg.lstsq(numpy.column_stack([v_sep, v_align, v_cohes,v_obst, numpy.ones(v_sep.shape[0:1])]), b[:,1])
	z_res = numpy.linalg.lstsq(numpy.column_stack([v_sep, v_align, v_cohes,v_obst, numpy.ones(v_sep.shape[0:1])]), b[:,2])
	return x_res, y_res, z_res

def test():
	sigmas = ("0.1","0.5","1.0")
	combined_results = dict()
	independent_results = dict()
	results_file = open("results.p","w")
	for sep in sigmas:
		s = "rbfsepvec-"+sep+"sigma.btf"
		for ori in sigmas:
			o = "rbforivec-"+ori+"sigma.btf"
			for coh in sigmas:
				c = "rbfsepvec-"+coh+"sigma.btf"
				print "Parameter settings:",(sep,ori,coh)
				print "Combined."
				combined_results[(s,o,c)] = computeLRcombined(s,o,c,"dvel.btf")
				print "Independent."
				independent_results[(s,o,c)] = computeLRindependent(s,o,c,"dvel.btf")
	print "Saving results to", results_file
	cPickle.dump({"combined":combined_results,"independent":independent_results},results_file)
	results_file.close()

def report_best(results):
	combined_results = results["combined"]
	independent_results = results["independent"]
	sigmas = ("0.1","0.5","1.0")
	min_combined = None
	min_combined_parameters = None
	min_independent = None
	min_independent_parameters = None
	for sep in sigmas:
		s = "rbfsepvec-"+sep+"sigma.btf"
		for ori in sigmas:
			o = "rbforivec-"+ori+"sigma.btf"
			for coh in sigmas:
				c = "rbfsepvec-"+coh+"sigma.btf"
				tmp = combined_results[(s,o,c)][1].sum()
				if tmp == 0:
					continue
				if (min_combined is None) or (tmp < min_combined):
					min_combined = tmp
					min_combined_parameters = (s,o,c)
	print "Min residual:", min_combined
	print "Parameters:", min_combined_parameters