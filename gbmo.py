import numpy,scipy.optimize,scipy.stats,btfutil,stats,time,os,os.path,tempfile,cPickle,sys,subprocess

def nullmin(fun,x0,args,**kwargs):
	"""
	A null local minimization function.

	For use with scipy.optimize.minimize and friends, this function performs no
	minimization, and exactly 1 evaluation. Pretty much only useful when function
	evaluations are very expensive and random search is the best you can do
	"""
	return scipy.optimize.OptimizeResult({'x':x0,'success':True,'fun':fun(x0)})

def evaluate_sim(model,num_steps,behav_measures,lr_shape,eps,tdir):
	# tdir = tempfile.mkdtemp()
	outname = os.path.join(tdir,'lr_coeff.txt')
	prefix = "[BTFLogger] Starting new logs in"
	outf = open(outname,"w")
	model = model.reshape(lr_shape)
	for row in model:
		outf.write("%f %f %f\n"%(row[0],row[1],row[2]))
	outf.close()
	proc = subprocess.Popen(['java','biosim.app.fishlr.FishLR','-nogui','-logging',tdir,'-lr',outname,'-for',str(num_steps)],stdout=subprocess.PIPE)
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
		print "Computing",key,"timeseries for [simulated]"
		sim_ts = btfutil.timeseries(sim_btf,key,pose_cnames)
		print "Computing histogram"
		sim_hist = numpy.histogram(sim_ts[1],bins=behav_measures[key][1])
		sim_hist_normed=sim_hist[0]/float(sim_hist[0].sum())
		rv += scipy.stats.entropy(sim_hist[0]+eps,behav_measures[key][0]+eps)
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
	return scipy.optimize.basinhopping(evaluate_sim,initial_guess,niter=niter,minimizer_kwargs={"method":"L-BFGS-B","args":(numsteps,behav_measures_dict,initial_guess.shape,0.000001,tdir),"options":{"maxfun":maxfun}},disp=True)
	# so with an x with 9 spots, this should run for about 336 iterations.

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
	measures = [stats.maxDist,stats.avgNNDist,stats.varNNDist]
	print "Optimizing"
	result = optimize(gen_btf,numsteps,measures,9,niter=niter,maxfun=maxfun)
	print "Saving result to", sys.argv[4]
	cPickle.dump(result,open(sys.argv[4],"w"))
	print "Done!"
