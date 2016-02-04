import numpy, btfutil, scipy.spatial, subprocess, time, sys, cPickle, os, os.path, tempfile, multiprocessing, random, pandas

class KNN():
	def __init__(self,features,ys):
		self.kdt = scipy.spatial.cKDTree(features)
		self.ys = ys
	def query(self,features,k):
		self.ys[self.kdt.query(features,k)[1]]
	def to_csv(self,outf,feature_names):
		data_df = pandas.DataFrame(numpy.column_stack(self.kdt.data,self.ys),columns=feature_names)
		data_df.to_csv(outf,index=False)

def writeInitialPlacement(outf,initialPlacementBTF):
	rowIdx = 0
	while rowIdx < len(initialPlacementBTF['id']) and initialPlacementBTF['clocktime'][rowIdx] == initialPlacementBTF['clocktime'][0]:
		outf.write(initialPlacementBTF['id'][rowIdx])
		outf.write(" "+initialPlacementBTF['xpos'][rowIdx])
		outf.write(" "+initialPlacementBTF['ypos'][rowIdx])
		outf.write(" "+initialPlacementBTF['timage'][rowIdx]+"\n")
		rowIdx += 1

def predictLR(model,num_steps,initialPlacementBTF,logdir=None,feature_names=None):
	if logdir is None:
		logdir = os.getcwd()
	
	outname = os.path.join(logdir,'lr_coeff.txt')
	prefix = "[BTFLogger] Starting new logs in"
	outf = open(outname,'w')
	for row in model:
		outf.write("%f %f %f\n"%(row[0],row[1],row[2]))
	outf.close()
	outf = open(os.path.join(logdir,"initial_placement.txt"),"w")
	# rowIdx = 0
	# while rowIdx < len(initialPlacementBTF['id']) and initialPlacementBTF['clocktime'][rowIdx] == initialPlacementBTF['clocktime'][0]:
	# 	outf.write(initialPlacementBTF['id'][rowIdx])
	# 	outf.write(" "+initialPlacementBTF['xpos'][rowIdx])
	# 	outf.write(" "+initialPlacementBTF['ypos'][rowIdx])
	# 	outf.write(" "+initialPlacementBTF['timage'][rowIdx]+"\n")
	# 	rowIdx += 1
	# outf.close()
	writeInitialPlacement(outf,initialPlacementBTF)
	outf.close()
	# proc = subprocess.Popen(['java','biosim.app.fishlr.FishLR','-placed','-btf',initialPlacementBTFDir,'-nogui','-logging', '-lr', outname,'-for',str(num_steps)],stdout=subprocess.PIPE)
	proc = subprocess.Popen(['java','biosim.app.fishlr.FishLR','-placed', os.path.join(logdir,'initial_placement.txt'),'-nogui','-logging', logdir, '-lr', outname,'-for',str(num_steps)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
	output,errors = proc.communicate()
	trace_btfdir_start = len(prefix)+output.index(prefix)
	trace_btfdir_end = output.index("\n",trace_btfdir_start)
	trace_btfdir = output[trace_btfdir_start:trace_btfdir_end].strip()
	rv = btfutil.BTF()
	rv.import_from_dir(trace_btfdir)
	rv.filter_by_col('dbool')
	return rv

def learnLR(features,ys,cv_features=None,cv_ys=None):
	result = numpy.linalg.lstsq(features,ys)
	if not((cv_features is None) or (cv_ys is None)):
		print "CV error:",numpy.linalg.norm(cv_ys - (cv_features.dot(result[0])))
	return result[0]

def predict_KNN(model, num_steps, initialPlacementBTF,logdir=None,feature_names=None):
	if logdir is None:
		logdir = os.getcwd()
	outname = os.path.join(logdir,'knn_dataset.csv')
	prefix = "[BTFLogger] Starting new logs in"
	outf = open(outname,'w')
	model.to_csv(outf,feature_names)
	outf.close()
	placementFname = os.path.join(logdir,'initial_placement.txt')
	outf = open(placementFname,'w')
	writeInitialPlacement(outf,initialPlacementBTF)
	outf.close()
	proc = subprocess.Popen(['java',\
							'biosim.app.fishreynolds.FishReynods',\
							'-placed',placementFname,\
							'-nogui',\
							'-logging',logdir,\
							'-lr', outname,\
							'-for',str(num_steps)],\
							stdout = subprocess.PIPE, stderr=subprocess.PIPE)
	output,errors = proc.communicate()
	trace_btfdir_start = len(prefix)+output.index(prefix)
	trace_btfdir_end = output.index('\n',trace_btfdir_start)
	trace_btfdir = output[trace_btfdir_start:trace_btfdir_end].strip()
	rv = btfutil.BTF()
	rv.import_from_dir(trace_btfdir)
	rv.filter_by_col('dbool')
	return rv

def learnKNN(features,ys):
	return KNN(features,ys)

def btf2data(btf,feature_names,augment):
	features = numpy.column_stack([map(lambda line: map(float,line.split()), btf[col_name]) for col_name in feature_names])
	if augment:
		features = numpy.column_stack([features,numpy.ones(features.shape[0])])
	ys = numpy.array(map(lambda line: map(float, line.split()), btf['dvel']))
	return features,ys

def split_btf_trajectory(btf,feature_names,augment):
	features,ys = btf2data(btf,feature_names,augment)
	npid = numpy.array(map(int,btf['id']))
	unique_ids = set(npid)
	return {eyed:features[npid==eyed] for eyed in unique_ids}

def dad(N,k,training_dir,learn,predict,feature_names = ['rbfsepvec','rbforivec','rbfcohvec','rbfwallvec'], weight_dad_samples=None):
	btf = btfutil.BTF()
	btf.import_from_dir(training_dir)
	btf.filter_by_col('dbool')
	features, ys = btf2data(btf,feature_names,augment=True)
	tmpidx = int(features.shape[0]*0.8)
	training_features = features[:tmpidx,:]
	cv_features = features[tmpidx:,:]
	training_ys = ys[:tmpidx,:]
	cv_ys = ys[tmpidx:,:]
	trajectory = split_btf_trajectory(btf,['xpos','ypos','timage'],augment=False)
	training_trajectory = {eyed:trajectory[eyed][:tmpidx,:] for eyed in trajectory}
	cv_trajectory = {eyed:trajectory[eyed][tmpidx:,:] for eyed in trajectory}
	# min_seq_length = min(map(lambda key: training_trajectory[key].shape[0],training_trajectory)) 
	# predict_steps = min_seq_length
	# if k > 0:
	# 	predict_steps = min(min_seq_length,k)
	# print "k=",predict_steps
	# training_traj_features = split_btf_trajectory(btf,feature_names)
	models = (learn(training_features,training_ys),)
	dad_training_features, dad_training_ys = None, None
	for n in range(N):
		sim_btf = predict(models[n],k,training_dir)
		sim_features,sim_ys = btf2data(sim_btf,feature_names,augment=True)
		sim_trajectory = split_btf_trajectory(sim_btf,['xpos','ypos','timage'],augment=False)
		sim_traj_features = split_btf_trajectory(sim_btf,feature_names,augment=True)
		if dad_training_features is None:
			dad_training_features, dad_training_ys = training_features, training_ys
		for eyed in sim_trajectory:
			traj = sim_trajectory[eyed]
			traj_feats = sim_traj_features[eyed]
			for row_idx in range(1,min(training_trajectory[eyed].shape[0]-1,traj.shape[0]-1)):
				dad_sample_feats = traj_feats[row_idx]
				dad_sample_ys = training_trajectory[eyed][row_idx+1]-traj[row_idx]
				dad_training_features = numpy.row_stack([dad_training_features,dad_sample_feats])
				dad_training_ys = numpy.row_stack([dad_training_ys, dad_sample_ys])
		if not(weight_dad_samples is None):
			if weight_dad_samples == 1:
				weight_dad_samples = max(1,int(ys.shape[0]/dad_training_ys.shape[0]))
			print "Reweighting samples:",weight_dad_samples
			dad_training_features = numpy.row_stack([dad_training_features,]*weight_dad_samples)
			dad_training_ys = numpy.row_stack([dad_training_ys,]*weight_dad_samples)
		models = models + (learn(dad_training_features,dad_training_ys,cv_features,cv_ys),)
	return models

def dad_subseq(N,k,training_btf_tuple,learn,predict,feature_names=['rbfsepvec','rbforivec','rbfcohvec','rbfwallvec'],savetofile=False,fixed_data_ratio=False):
	training_features,training_ys = list(),list()
	#randomize sample order
	random.shuffle(training_btf_tuple)
	cutoff = int(len(training_btf_tuple)*0.8)
	cv_tuple = training_btf_tuple[cutoff:]
	training_btf_tuple = training_btf_tuple[:cutoff]
	training_trajectories = list()
	logdir = tempfile.mkdtemp(suffix='_dad',prefix='logging_',dir=os.getcwd())
	print "Logging to",logdir
	for btf in training_btf_tuple:
		f,y = btf2data(btf,feature_names,augment=True)
		training_features.append(f)
		training_ys.append(y)
		#if training_features is None:
		#	training_features = f
		#	training_ys = y
		#else:
		#	training_features = numpy.row_stack([training_features,f])
		#	training_ys = numpy.row_stack([training_ys,y])
		training_trajectories.append(split_btf_trajectory(btf,['xpos','ypos','timage'],augment=False))
	cv_features, cv_ys = None,None
	for cv_btf in cv_tuple:
		if cv_features is None:
			cv_features,cv_ys = btf2data(cv_btf,feature_names,augment=True)
		else:
			tmpF, tmpY = btf2data(cv_btf,feature_names,augment=True)
			cv_features = numpy.row_stack([cv_features,tmpF])
			cv_ys = numpy.row_stack([cv_ys,tmpY])
	#models = (learn(training_features,training_ys),)
	#dad_training_features, dad_training_ys = training_features, training_ys
	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	num_tracklet_samples = list()
	reserve_tuple_size = None
	if fixed_data_ratio:
		print 'computing initial data size'
		btf_len_sum = 0
		for btf in training_btf_tuple:
			nm_samples = len(btf['id'])
			num_tracklet_samples.append(nm_samples)
			btf_len_sum += nm_samples
		# print "initial data size:", btf_len_sum, '/', (2**N), '=', btf_len_sum/float(2**N)
		init_num_tuples = min(4,multiprocessing.cpu_count())
		print 'initial num tuples:',init_num_tuples
		print "Max iterations:", numpy.log(btf_len_sum/float(init_num_tuples))/numpy.log(2)
		#btf's
		reserve_btf_tuple = training_btf_tuple[init_num_tuples:]
		training_btf_tuple = training_btf_tuple[:init_num_tuples]
		#trajectories
		reserve_trajectories = training_trajectories[init_num_tuples:]
		training_trajectories = training_trajectories[:init_num_tuples]
		#num samples
		reserve_tuple_size = num_tracklet_samples[init_num_tuples:]
		num_tracklet_samples = num_tracklet_samples[:init_num_tuples]
		#initial training sets
		reserve_training_features = training_features[init_num_tuples:]
		training_features = training_features[:init_num_tuples]
		reserve_training_ys = training_ys[init_num_tuples:]
		training_ys = training_ys[:init_num_tuples]
		print 'Initial samples:',sum(num_tracklet_samples)
		print 'Reserved samples:', sum(reserve_tuple_size)
	models = (learn(numpy.row_stack(training_features),numpy.row_stack(training_ys)),)
	#dad_training_features, dad_training_ys = training_features, training_ys
	dad_training_features, dad_training_ys, num_dad_samples = list(), list(), sum(num_tracklet_samples)
	for n in range(N):
		print "Iteration",n
		results = pool.map(multiproc_hack,args_generator(training_btf_tuple,training_trajectories,predict,models[n],k,logdir,feature_names,n))
		new_feats, new_ys = pool.map(numpy.row_stack,zip(*results))
		# new_feats, new_ys = pool.map(numpy.row_stack,zip(*results))
		#dad_training_features = numpy.row_stack([dad_training_features,new_feats])
		dad_training_features.append(new_feats)
		#dad_training_ys = numpy.row_stack([dad_training_ys,new_ys])
		dad_training_ys.append(new_ys)
		#nm_dad_samples = len(dad_training_ys)
		num_dad_samples += len(new_ys)
		print "num dad samples:",num_dad_samples
		if fixed_data_ratio:
			added_tuple_size = 0
			while (num_dad_samples > added_tuple_size) and (len(reserve_trajectories) > 0):
				training_btf_tuple.append(reserve_btf_tuple.pop())
				training_trajectories.append(reserve_trajectories.pop())
				num_tracklet_samples.append(reserve_tuple_size.pop())
				added_tuple_size += num_tracklet_samples[-1]
				training_features.append(reserve_training_features.pop())
				training_ys.append(reserve_training_ys.pop())
		models = models + (learn(numpy.row_stack(dad_training_features+training_features),\
					 numpy.row_stack(dad_training_ys+training_ys),cv_features,cv_ys),)
		if fixed_data_ratio:
			if not(len(reserve_trajectories) > 0):
				print "Ran out of data after iteration", n
				break
	if savetofile:
		picklename = os.path.join(logdir,'dad-subseq-results.p')
		print "Saving models to [%s]"%(picklename,)
		cPickle.dump(models,open(picklename,'w'))
	return models

def args_generator(training_btf_tuple, training_trajectories,predict,model,k,logdir,feature_names,iteration):
	for idx in range(len(training_btf_tuple)):
		# logdir = tempfile.mkdtemp(suffix='_dad',prefix='logging_',dir=os.getcwd())
		new_logdir = tempfile.mkdtemp(suffix='_seq_%d'%idx,prefix='it_%d_'%iteration,dir=logdir)
		yield (training_btf_tuple[idx],training_trajectories[idx],predict,model,k,new_logdir,feature_names)

def multiproc_hack(args):
	return do_subseq_inner_loop(args[0],args[1],args[2],args[3],args[4],args[5],args[6])

def do_subseq_inner_loop(subseqBTF,training_trajectory,predict,model,k,logdir,feature_names):
	sim_btf = predict(model,k,subseqBTF,logdir,["sepX", "sepY","oriX","oriY","cohX","cohY","wallX","wallY","dvelX","dvelY","dvelT"])
	sim_features, sim_ys = btf2data(sim_btf, feature_names, augment=True)
	sim_trajectory = split_btf_trajectory(sim_btf,['xpos','ypos','timage'], augment=False)
	sim_traj_features = split_btf_trajectory(sim_btf, feature_names, augment=True)
	feats_rv = list()
	ys_rv = list()
	for eyed in sim_trajectory:
		traj = sim_trajectory[eyed]
		traj_feats = sim_traj_features[eyed]
		num_dad_samples = min(training_trajectory[eyed].shape[0],traj.shape[0])-1
		traj_feats_rv = numpy.zeros((num_dad_samples,traj_feats.shape[1]))
		traj_ys_rv = numpy.zeros((num_dad_samples,traj.shape[1]))
		# for row_idx in range(1,min(training_trajectory[eyed].shape[0]-1,traj.shape[0]-1)):
		for row_idx in range(num_dad_samples):
			traj_feats_rv[row_idx] = traj_feats[row_idx]
			traj_ys_rv[row_idx] = training_trajectory[eyed][row_idx+1]-traj[row_idx]
		feats_rv.append(traj_feats_rv)
		ys_rv.append(traj_ys_rv)
	return numpy.row_stack(feats_rv), numpy.row_stack(ys_rv)

def find_best_model(training_dir,model_list,feature_names=['rbfsepvec','rbforivec','rbfcohvec','rbfwallvec']):
	btf = btfutil.BTF()
	btf.import_from_dir(training_dir)
	btf.filter_by_col('dbool')
	features, ys = btf2data(btf,feature_names,augment=True)
	errors = map(lambda m: numpy.linalg.norm(ys-features.dot(m)), model_list)
	rv_min = min(errors)
	rv_idx = errors.index(rv_min)
	print "Minimum error:",rv_min
	print "Iteration:",rv_idx
	return model_list[rv_idx]

def subseqmain(subseq_fname, num_models, max_seq_len):
	print "loading btfs from",subseq_fname
	btf_tuple = list(cPickle.load(open(subseq_fname)))
	models = dad_subseq(num_models,max_seq_len,btf_tuple,learnLR,predictLR, savetofile=True, fixed_data_ratio=True)
	#cPickle.dump(models,open("dad-subseq-results.p","w"))

def main(training_dir,num_models,max_seq_len):
	models = dad(num_models,max_seq_len,training_dir,learnLR,predictLR)
	# print models
	cPickle.dump(models,open("dad-results.p","w"))

if __name__ == '__main__':
	if len(sys.argv) == 4:
		# main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
		subseqmain(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
	elif len(sys.argv) == 3:
		model_list = cPickle.load(open(sys.argv[2]))
		best_model = find_best_model(sys.argv[1],model_list)
		outname = 'dad-lr_coeff.txt'
		print "Saving as", outname
		outf = open(outname,'w')
		for row in best_model:
			outf.write("%f %f %f\n"%(row[0],row[1],row[2]))
		outf.close()
