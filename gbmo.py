import numpy,scipy.optimize,scipy.stats,btfutil,stats,time,os,os.path,tempfile,cPickle,sys,subprocess,cma
import matplotlib
import multiprocessing

# matplotlib.pyplot.ion()

EPS=0.00000000001

GEN_CTR=0

def nullmin(fun,x0,args,**kwargs):
	"""
	A null local minimization function.

	For use with scipy.optimize.minimize and friends, this function performs no
	minimization, and exactly 1 evaluation. Pretty much only useful when function
	evaluations are very expensive and random search is the best you can do
	"""
	return scipy.optimize.OptimizeResult({'x':x0,'success':True,'fun':fun(x0)})

# def mychoice(seq,p,size=1):
# 	samples = numpy.random.multinomial(size,pvals=p)
# 	return reduce(lambda s,x: s+x, [[seq[idx],]*samples[idx] for idx in range(len(seq))])
def gen_gauss_eval2(num_m, numSamples,given=None):
	mix_probs = numpy.array([1.0/float(num_m),]*int(num_m)) #mix_probs = numpy.random.dirichlet([1.0/float(num_m),]*int(num_m))
	if given is None:
		means = numpy.random.random(num_m)*15.0
		sigmas = numpy.random.random(num_m)*5.0
	else:
		means=given[:int(num_m)]
		sigmas=given[int(num_m):int(num_m*2)]
	sample_choices = numpy.random.choice(range(num_m),size=numSamples,p=mix_probs)
	gen_samples = [numpy.random.normal(means[thing],sigmas[thing]) for thing in sample_choices]
	gen_hist = numpy.histogram(gen_samples,bins=50)
	gen_hist_normed = gen_hist[0]/float(gen_hist[0].sum())
	def rv(x,disp=False):
		tmp_mix_rates = numpy.array([1.0/float(num_m),]*int(num_m)) #x[:num_m]
		tmp_means = x[:num_m]
		tmp_sigmas = x[num_m:2*num_m]
		tmp_sc = numpy.random.choice(range(num_m),size=numSamples,p=tmp_mix_rates)
		tmp_samples = [numpy.random.normal(tmp_means[thing],max(EPS,tmp_sigmas[thing])) for thing in tmp_sc]
		sim_hist = numpy.histogram(tmp_samples,bins=gen_hist[1])
		sim_hist_normed = sim_hist[0]/float(sim_hist[0].sum())
		if disp:
			matplotlib.pyplot.clf()
			matplotlib.pyplot.plot(gen_hist[1][:-1],gen_hist_normed)
			matplotlib.pyplot.plot(sim_hist[1][:-1],sim_hist_normed)
			matplotlib.pyplot.show()
		return scipy.stats.entropy(sim_hist_normed+0.000001,gen_hist_normed+0.000001)
	return rv,means,sigmas,mix_probs


def gen_gauss_eval(m1,m2,p1, numSamples):
	foo = numpy.array([numpy.random.normal(m1,1.0) if i < p1 else numpy.random.normal(m2,1.0) for i in numpy.random.random(numSamples)])
	def rv(x,disp=False):
		bar = numpy.array([numpy.random.normal(x[0],1.0) if i < x[2] else numpy.random.normal(x[1],1.0) for i in numpy.random.random(numSamples)])
		gen_hist = numpy.histogram(foo,bins=50)
		gen_hist_normed = gen_hist[0]/float(gen_hist[0].sum())
		sim_hist = numpy.histogram(bar,bins=gen_hist[1])
		sim_hist_normed = sim_hist[0]/float(sim_hist[0].sum())
		if disp:
			matplotlib.pyplot.clf()
			matplotlib.pyplot.plot(gen_hist[1][:-1],gen_hist_normed)
			matplotlib.pyplot.plot(sim_hist[1][:-1],sim_hist_normed)
			matplotlib.pyplot.show()
		return scipy.stats.entropy(sim_hist_normed+0.000001,gen_hist_normed+0.000001)
	return rv

def evaluate_sim(model,num_steps,behav_measures,lr_shape,eps,tdir):
	# tdir = tempfile.mkdtemp()
	new_logdir = tempfile.mkdtemp(prefix='generation_%d_candidate_'%GEN_CTR,dir=tdir)
	outname = os.path.join(new_logdir,'lr_coeff.txt')
	prefix = "[BTFLogger] Starting new logs in"
	outf = open(outname,"w")
	model = model.reshape(lr_shape)
	for row in model:
		outf.write("%f %f %f\n"%(row[0],row[1],row[2]))
	outf.close()
	proc = subprocess.Popen(['java','biosim.app.fishlr.FishLR','-nogui','-logging',new_logdir,'-lr',outname,'-for',str(num_steps)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
	output,errors = proc.communicate()
	trace_btfdir_start = len(prefix)+output.index(prefix)
	trace_btfdir_end = output.index("\n",trace_btfdir_start)
	trace_btfdir = output[trace_btfdir_start:trace_btfdir_end].strip()
	sim_btf = btfutil.BTF()
	sim_btf.import_from_dir(trace_btfdir)
	#sim_btf.filter_by_col('dbool')
	#Ok, now compute the histograms of the behavior measures
	pose_cnames = ['xpos','ypos','timage']
	rv = 0.0
	for key in behav_measures:
		# print "Computing",key,"timeseries for [simulated]"
		sim_ts = btfutil.timeseries(sim_btf,key,pose_cnames)
		# print "Computing histogram"
		sim_hist = numpy.histogram(sim_ts[1],bins=behav_measures[key][1])
		sim_hist_normed=sim_hist[0]/float(sim_hist[0].sum())
		rv += scipy.stats.entropy(sim_hist_normed+eps,behav_measures[key][0]+eps)
	return rv

def optimize(btf, numsteps, behavem_list,initial_guess,bins=50,maxfun=30,niter=5):
	#initialize a random starting place if one isn't provided
	if type(initial_guess) == int:
		initial_guess = numpy.random.random((initial_guess,3))
	#get generating behavior measures
	tdir = tempfile.mkdtemp(prefix="logs_gbmo_",dir=os.getcwd())
	print "logging dir:",tdir
	behav_measures_dict = dict()
	pose_cnames = ['xpos','ypos','timage']
	for item in behavem_list:
		print "Computing", item, "timeseries for [generating]"
		gen_ts = btfutil.timeseries(gen_btf,item,pose_cnames)
		print "Computing histogram"
		gen_hist = numpy.histogram(gen_ts[1],bins=bins)
		gen_hist_normed =gen_hist[0]/float(gen_hist[0].sum())
		behav_measures_dict[item] = (gen_hist_normed,gen_hist[1])
	# start optimizing
	# res = ("basinhopping",)+scipy.optimize.basinhopping(evaluate_sim,initial_guess,niter=niter,minimizer_kwargs={"method":"L-BFGS-B","args":(numsteps,behav_measures_dict,initial_guess.shape,0.000001,tdir),"options":{"maxfun":maxfun}},disp=True)
	# res = ("anneal",)+scipy.optimize.anneal(evaluate_sim,initial_guess,args=(numsteps,behav_measures_dict,initial_guess.shape,0.000001,tdir),lower=-25.0,upper=1.0,maxeval=(maxfun*niter),full_output=True)
	tmp_opts = dict()
	# tmp_opts['args'] = (numsteps,behav_measures_dict,initial_guess.shape,0.000001,tdir)
	global global_args
	global_args['numsteps'] = numsteps
	global_args['behav_measures_dict'] = behav_measures_dict
	global_args['lr_shape'] = initial_guess.shape
	global_args['eps'] = 0.000001
	global_args['tdir'] = tdir
	tmp_opts["bounds"] = [-25.0,1.0]
	tmp_opts["maxfevals"] = (niter*maxfun)
	es = cma.CMAEvolutionStrategy(x0=initial_guess.reshape((-1,)), sigma0=1.0, inopts=tmp_opts)
	# SINGLE PROCESS
	# cma_res = cma.fmin(evaluate_sim,initial_guess.reshape((-1,)),1.0,args=(numsteps,behav_measures_dict,initial_guess.shape,0.000001,tdir),options={"bounds":[-25.0,1.0],"maxfevals":(niter*maxfun)})
	
	# MULTIPROCESSING
	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	global GEN_CTR
	GEN_CTR=0
	while not(es.stop()):
		print "Generation", GEN_CTR
		start_time = time.time()
		candidates = es.ask()
		print "Num candidates:",len(candidates)
		evls = pool.map(eval_wrapper,candidates)
		es.tell(candidates,evls)
		end_time = time.time()
		es.disp()
		GEN_CTR=GEN_CTR+1
		print "Generation took",end_time-start_time,"seconds"
	cma_res = es.result()

	#cma_res[-1].load()
	#res = ("cma",)+cma_res[:-3]+(cma_res[-1].f[:,[1,4,5]],)
	res = ("cma",)+cma_res
	return res

global_args = {}

def eval_wrapper(model):
	return evaluate_sim(model,global_args['numsteps'],global_args['behav_measures_dict'],global_args['lr_shape'],global_args['eps'],global_args['tdir'])

if __name__ == '__main__':
	gen_btf = btfutil.BTF()
	maxfun=30
	niter=5
	if len(sys.argv)>5:
		niter=int(sys.argv[5])
		if len(sys.argv)>6:
			maxfun = int(sys.argv[6])
	print "gen behavior dir:",sys.argv[1]
	gen_btf.import_from_dir(sys.argv[1])
	print "num simulation steps:",sys.argv[2]
	numsteps = int(sys.argv[2])
	print "num features:",sys.argv[3]
	num_features = int(sys.argv[3])
	measures = [stats.maxDist,stats.avgNNDist,stats.varNNDist,stats.polarizationOrder,stats.rotationOrder]
	print "Optimizing"
	result = optimize(gen_btf,numsteps,measures,9,niter=niter,maxfun=maxfun)
	print "Saving result to", sys.argv[4]
	cPickle.dump(result,open(sys.argv[4],"w"))
	print "Done!"
