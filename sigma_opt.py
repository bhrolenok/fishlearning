# sigma_opt.py

import numpy,scipy.optimize,time,sys,math,cPickle

DATA_neighbor_x = None
DATA_neighbor_y = None
DATA_neighbor_ori = None
DATA_obs_vec = None
DATA_b = None
num_evals = 0
memoized = {}

wait_time = 5.0 # in seconds

DEBUG_MAXLINES = -1

def load_data(nvec_fname, nori_fname, obs_fname, b_fname):
	global DATA_neighbor_x, DATA_neighbor_y, DATA_neighbor_ori, DATA_obs_vec, DATA_b
	DATA_neighbor_y = list()
	DATA_neighbor_x = list()
	DATA_neighbor_ori = list()
	DATA_obs_vec = list()
	DATA_b = list()
	print "Loading",nvec_fname
	lines = open(nvec_fname).readlines()[:DEBUG_MAXLINES]
	num_lines = len(lines)
	last_checked = time.time()
	last_line_idx = 0
	for idx in range(num_lines):
		line = map(float,lines[idx].strip().split())
		x = numpy.array(line[::2])
		y = numpy.array(line[1::2])
		DATA_neighbor_x.append(x)
		DATA_neighbor_y.append(y)
		cur_time = time.time()
		if (cur_time-last_checked) > wait_time:
			print "Line",idx,"lps",(idx-last_line_idx)/(cur_time-last_checked), 100*float(idx)/float(num_lines),"%"
			last_checked = cur_time
			last_line_idx = idx
	print "Loading", nori_fname
	lines = open(nori_fname).readlines()[:DEBUG_MAXLINES]
	num_lines = len(lines)
	last_checked = time.time()
	last_line_idx = 0
	for idx in range(num_lines):
		line = map(float,lines[idx].strip().split())
		DATA_neighbor_ori.append(line)
		cur_time = time.time()
		if (cur_time-last_checked) > wait_time:
			print "Line",idx,"lps",(idx-last_line_idx)/(cur_time-last_checked), 100*float(idx)/float(num_lines),"%"
			last_checked = cur_time
			last_line_idx = idx
	print "Loading",obs_fname
	DATA_obs_vec = numpy.array([map(float,l.split()) for l in open(obs_fname).readlines()[:DEBUG_MAXLINES]])
	print "Loading",b_fname
	DATA_b = numpy.array([map(float,l.split()) for l in open(b_fname).readlines()[:DEBUG_MAXLINES]])
	print "Done loading"

def filter_dbool(dbool_fname):
	global DATA_b, DATA_obs_vec, DATA_neighbor_x, DATA_neighbor_y, DATA_neighbor_ori
	print "Filtering from",dbool_fname
	filt = map(lambda x: x.strip() == 'True', open(dbool_fname).readlines()[:DEBUG_MAXLINES])
	print "filtering DATA_b"
	# DATA_b = DATA_b[filt]
	DATA_b = numpy.array(filter(lambda d: not(d is None), map(lambda foo,bar: bar if foo else None, filt, DATA_b)))
	print "filtering DATA_obs_vec"
	# DATA_obs_vec = DATA_obs_vec[filt]
	DATA_obs_vec = numpy.array(filter(lambda d: not(d is None), map(lambda foo,bar: bar if foo else None, filt, DATA_obs_vec)))
	print "filtering DATA_neighbor_x"
	DATA_neighbor_x = filter(lambda d: not(d is None), map(lambda foo,bar: bar if foo else None, filt, DATA_neighbor_x))
	print "filtering DATA_neighbor_y"
	DATA_neighbor_y = filter(lambda d: not(d is None), map(lambda foo,bar: bar if foo else None, filt, DATA_neighbor_y))
	print "filtering DATA_neighbor_ori"
	DATA_neighbor_ori = filter(lambda d: not(d is None), map(lambda foo,bar: bar if foo else None, filt, DATA_neighbor_ori))
	print "Done filtering"
	# print map(len, [DATA_neighbor_x, DATA_neighbor_y, DATA_neighbor_ori, DATA_obs_vec, DATA_b] )

def rotate(x,y,theta):
	x=float(x)
	y=float(y)
	theta=float(theta)
	xprime = (x*math.cos(theta))-(y*math.sin(theta))
	yprime = (x*math.sin(theta))+(y*math.cos(theta))
	return xprime,yprime

def weightedData(sigmas):
	sep_sigma, ori_sigma, coh_sigma, wall_sigma = sigmas
	num_lines = len(DATA_neighbor_x)
	xs = DATA_neighbor_x
	ys = DATA_neighbor_y

	sep_weights = map(lambda x,y: weights(x,y,sep_sigma), xs, ys)
	ori_weights = map(lambda x,y: weights(x,y,ori_sigma), xs, ys)
	coh_weights = map(lambda x,y: weights(x,y,coh_sigma), xs, ys)
	wall_weights = map(lambda x,y: weights(x,y,wall_sigma), DATA_obs_vec[:,0], DATA_obs_vec[:,1])

	sep = map(sep_vec,sep_weights,xs,ys)
	ori = map(ori_vec,ori_weights,DATA_neighbor_ori)
	coh = map(sep_vec,coh_weights,xs,ys)
	obs = map(lambda weight, x, y: (weight*x,weight*y),wall_weights,DATA_obs_vec[:,0],DATA_obs_vec[:,1])
	return sep, ori, coh, obs

def optimize(sigmas):
	global num_evals, memoized
	num_evals = num_evals+1
	print "Eval #"+str(num_evals), "Testing sigmas:",sigmas
	if tuple(sigmas) in memoized:
		return memoized[tuple(sigmas)]
	# sep_sigma, ori_sigma, coh_sigma, wall_sigma = sigmas
	# num_lines = len(DATA_neighbor_x)
	# xs = DATA_neighbor_x
	# ys = DATA_neighbor_y

	# sep_weights = map(weights,xs,ys,(sep_sigma,)*num_lines)
	# ori_weights = map(weights,xs,ys,(ori_sigma,)*num_lines)
	# coh_weights = map(weights,xs,ys,(coh_sigma,)*num_lines)
	# wall_weights = map(weights,DATA_obs_vec[:,0],DATA_obs_vec[:,1],(wall_sigma,)*num_lines)

	# sep = map(sep_vec,sep_weights,xs,ys)
	# ori = map(ori_vec,ori_weights,DATA_neighbor_ori)
	# coh = map(sep_vec,coh_weights,xs,ys)
	# obs = map(lambda weight, x, y: (weight*x,weight*y),wall_weights,DATA_obs_vec[:,0],DATA_obs_vec[:,1])
	sep, ori, coh, obs = weightedData(sigmas)

	b = DATA_b

	#numpy.linalg.lstsq(numpy.column_stack([v_sep,v_align,v_cohes,v_obst,numpy.ones(v_sep.shape[0:1])]), b)
	x, residuals, rank, s = numpy.linalg.lstsq(numpy.column_stack([sep,ori,coh,obs,numpy.ones((len(sep),1))]), b)
	memoized[tuple(sigmas)] = residuals.sum()
	return residuals.sum()


def sep_vec(weights,x,y):
	rv = numpy.sum(numpy.column_stack([x,y]).T*weights,axis=1)/float(len(x))
	return rv

def ori_vec(weights,theta):
	rv = numpy.sum(numpy.array([rotate(1,0,t) for t in theta]).T*weights,axis=1)/float(len(theta))
	return rv

def weights(x,y,sigma):
	# x = numpy.array(line[::2])
	# y = numpy.array(line[1::2])
	weights = numpy.exp(-(numpy.power(x,2)+numpy.power(y,2))/(2.0*numpy.power(sigma,2)))
	return weights

class bounds(object):
	def __init__(self):
		pass
	def __call__(self, **kwargs):
		x = kwargs["x_new"]
		return numpy.all(x<=[2.0,2.0,2.0,2.0]) and numpy.all(x >=[0.0,0.0,0.0,0.0])

def nullmin(fun,x0,args, **kwargs):
	return scipy.optimize.OptimizeResult(x=x0,success=True)

def second_order_poly(data):
	num_columns = data.shape[1]
	res = list()
	for eye in range(num_columns):
		for jay in range(num_columns):
			res.append(data[:,eye]*data[:,jay])
	return res


def runLR(sigmas):
	sep_sigma, ori_sigma, coh_sigma, wall_sigma = sigmas
	# num_lines = len(DATA_neighbor_x)
	# xs = DATA_neighbor_x
	# ys = DATA_neighbor_y

	# sep_weights = map(weights,xs,ys,(sep_sigma,)*num_lines)
	# ori_weights = map(weights,xs,ys,(ori_sigma,)*num_lines)
	# coh_weights = map(weights,xs,ys,(coh_sigma,)*num_lines)
	# wall_weights = map(weights,DATA_obs_vec[:,0],DATA_obs_vec[:,1],(wall_sigma,)*num_lines)

	# sep = map(sep_vec,sep_weights,xs,ys)
	# ori = map(ori_vec,ori_weights,DATA_neighbor_ori)
	# coh = map(sep_vec,coh_weights,xs,ys)
	# obs = map(lambda weight, x, y: (weight*x,weight*y),wall_weights,DATA_obs_vec[:,0],DATA_obs_vec[:,1])

	# b = DATA_b
	data_list = list(weightedData(sigmas))
	print "Adding 2nd order poly features"
	data_list += second_order_poly(numpy.column_stack(data_list))
	b = DATA_b
	# return numpy.linalg.lstsq(numpy.column_stack([sep,ori,coh,obs,numpy.ones((len(sep),1))]),b)
	data_list.append(numpy.ones(b.shape[0]))
	# print map(len,data_list)
	return numpy.linalg.lstsq(numpy.column_stack(data_list),b)

if __name__ == '__main__':
	# global DEBUG_MAXLINES
	# DEBUG_MAXLINES=200
	# print "DEBUG_MAXLINES=",DEBUG_MAXLINES
	load_data(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
	filter_dbool(sys.argv[5])
	# print "Optimizing sigmas"
	# bnds = bounds()
	# results = scipy.optimize.basinhopping(optimize,[0.1,1.5,1.5,0.1], niter=100)
	# print "Opt Results"
	# print results
	# cPickle.dump(results,open("sopt-results.p","w"))	
	# best_sigmas = results.x
	best_sigmas = numpy.array([0.1,1.5,1.5,1.5])
	print "Running LR on sigmas:",best_sigmas
	lr_results = runLR(best_sigmas)
	print "LR Results"
	print lr_results
	cPickle.dump(lr_results,open("lr-results.p","w"))

