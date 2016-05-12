import numpy, scipy.spatial, pandas
import random
import subprocess, multiprocessing
import time, sys, os, os.path, tempfile, tarfile, shutil
import cPickle
import btfutil, linreg, knn

def dad(N,k,training_dir,learn,predict,feature_names = ['rbfsepvec','rbforivec','rbfcohvec','rbfwallvec','pvel'], weight_dad_samples=None,feature_column_names=None):
	btf = btfutil.BTF()
	btf.import_from_dir(training_dir)
	btf.filter_by_col('dbool')
	features, ys = btfutil.btf2data(btf,feature_names,augment=(learn!=learnKNN))
	tmpidx = int(features.shape[0]*0.8)
	training_features = features[:tmpidx,:]
	cv_features = features[tmpidx:,:]
	training_ys = ys[:tmpidx,:]
	cv_ys = ys[tmpidx:,:]
	trajectory = btfutil.split_btf_trajectory(btf,['xpos','ypos','timage'],augment=False)
	training_trajectory = {eyed:trajectory[eyed][:tmpidx,:] for eyed in trajectory}
	cv_trajectory = {eyed:trajectory[eyed][tmpidx:,:] for eyed in trajectory}
	# min_seq_length = min(map(lambda key: training_trajectory[key].shape[0],training_trajectory)) 
	# predict_steps = min_seq_length
	# if k > 0:
	# 	predict_steps = min(min_seq_length,k)
	# print "k=",predict_steps
	# training_traj_features = split_btf_trajectory(btf,feature_names)
	models = (learn(training_features,training_ys,feature_column_names=feature_column_names),)
	dad_training_features, dad_training_ys = None, None
	for n in range(N):
		sim_btf = predict(models[n],k,training_dir)
		sim_features,sim_ys = btfutil.btf2data(sim_btf,feature_names,augment=(learn!=learnKNN))
		sim_trajectory = btfutil.split_btf_trajectory(sim_btf,['xpos','ypos','timage'],augment=False)
		sim_traj_features = btfutil.split_btf_trajectory(sim_btf,feature_names,augment=(learn!=learnKNN))
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
		models = models + (learn(dad_training_features,dad_training_ys,cv_features,cv_ys,feature_column_names=feature_column_names),)
	return models

def dad_subseq(N,k,training_btf_tuple,learn,predict,feature_names=['rbfsepvec','rbforivec','rbfcohvec','rbfwallvec','pvel'],savetofile=False,fixed_data_ratio=False,feature_column_names=None, max_threads='inf'):
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
		f,y = btfutil.btf2data(btf,feature_names,augment=(learn!=knn.learnKNN))
		training_features.append(f)
		training_ys.append(y)
		training_trajectories.append(btfutil.split_btf_trajectory(btf,['xpos','ypos','timage'],augment=False))
	cv_features, cv_ys = None,None
	for cv_btf in cv_tuple:
		if cv_features is None:
			cv_features,cv_ys = btfutil.btf2data(cv_btf,feature_names,augment=(learn!=knn.learnKNN))
		else:
			tmpF, tmpY = btfutil.btf2data(cv_btf,feature_names,augment=(learn!=knn.learnKNN))
			cv_features = numpy.row_stack([cv_features,tmpF])
			cv_ys = numpy.row_stack([cv_ys,tmpY])
	pool = multiprocessing.Pool(min(multiprocessing.cpu_count(),max_threads))
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
	models = (learn(numpy.row_stack(training_features),numpy.row_stack(training_ys),feature_column_names=feature_column_names),)
	dad_training_features, dad_training_ys, num_dad_samples = list(), list(), sum(num_tracklet_samples)
	for n in range(N):
		print "Iteration",n
		results = pool.map(multiproc_hack,args_generator(training_btf_tuple,training_trajectories,predict,models[n],k,logdir,feature_names,n))
		new_feats, new_ys = pool.map(numpy.row_stack,zip(*results))
		dad_training_features.append(new_feats)
		dad_training_ys.append(new_ys)
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
					 numpy.row_stack(dad_training_ys+training_ys),cv_features,cv_ys,feature_column_names=feature_column_names),)
		if savetofile:
			picklename = os.path.join(logdir,'dad-subseq-results.p')
			print "Saving models to [%s]"%(picklename,)
			cPickle.dump(models,open(picklename,'w'))
		if fixed_data_ratio:
			if not(len(reserve_trajectories) > 0):
				print "Ran out of data after iteration", n
				break
	return models

def args_generator(training_btf_tuple, training_trajectories,predict,model,k,logdir,feature_names,iteration):
	for idx in range(len(training_btf_tuple)):
		new_logdir = tempfile.mkdtemp(suffix='_seq_%d'%idx,prefix='it_%d_'%iteration,dir=logdir)
		yield (training_btf_tuple[idx],training_trajectories[idx],predict,model,k,new_logdir,feature_names)

def multiproc_hack(args):
	try:
		return do_subseq_inner_loop(args[0],args[1],args[2],args[3],args[4],args[5],args[6])
	except:
		raise Exception("".join(traceback.format_exception(*sys.exc_info())))

def do_subseq_inner_loop(subseqBTF,training_trajectory,predict,model,k,logdir,feature_names):
	btflist = predict(model,k,subseqBTF,logdir)
	sim_trajectory, sim_traj_features = dict(), dict()
	for sim_btf in btflist:
		sim_features, sim_ys = btfutil.btf2data(sim_btf, feature_names, augment=(predict!=knn.predictKNN))
		tmp_traj = btfutil.split_btf_trajectory(sim_btf,['xpos','ypos','timage'], augment=False)
		for blerp in tmp_traj.keys():
			if blerp in sim_trajectory.keys():
				print "ERROR! ERROR! SOMETHING HAS GONE HORRIBLY AWRY!"
				raise RuntimeError("multiple predicted trajectories with same ID, don't know what to do, dying.")
		sim_trajectory.update(tmp_traj)
		tmp_traj_features =  btfutil.split_btf_trajectory(sim_btf, feature_names, augment=(predict!=knn.predictKNN))
		sim_traj_features.update(tmp_traj_features)
	# sim_trajectory = btfutil.split_btf_trajectory(sim_btf,['xpos','ypos','timage'], augment=False)
	# sim_traj_features = btfutil.split_btf_trajectory(sim_btf, feature_names, augment=(predict!=knn.predictKNN))
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

def find_best_model(training_dir,model_list,feature_names=['rbfsepvec','rbforivec','rbfcohvec','rbfwallvec','pvel'],use_augment=True):
	btf = btfutil.BTF()
	btf.import_from_dir(training_dir)
	btf.filter_by_col('dbool')
	features, ys = btfutil.btf2data(btf,feature_names,augment=use_augment)
	errors = map(lambda m: numpy.linalg.norm(ys-features.dot(m)), model_list)
	rv_min = min(errors)
	rv_idx = errors.index(rv_min)
	print "Minimum error:",rv_min
	print "Iteration:",rv_idx
	return model_list[rv_idx]

def subseqmain(subseq_fname, num_models, max_seq_len,feature_column_names=None):
	print "loading btfs from",subseq_fname
	btf_tuple = list(cPickle.load(open(subseq_fname)))
	#models = dad_subseq(num_models,max_seq_len,btf_tuple,linreg.learnLR,linreg.predictLR, savetofile=True, fixed_data_ratio=True)
	models = dad_subseq(num_models,max_seq_len,btf_tuple,knn.learnKNN,knn.predictKNN, savetofile=True, fixed_data_ratio=True,feature_column_names=feature_column_names,max_threads=4)

def main(training_dir,num_models,max_seq_len,feature_column_names=None):
	models = dad(num_models,max_seq_len,training_dir,linreg.learnLR,linreg.predictLR,feature_column_names=feature_column_names)
	# print models
	cPickle.dump(models,open("dad-results.p","w"))

if __name__ == '__main__':
	if len(sys.argv) == 4:
		# main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
		subseqmain(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),\
			feature_column_names=[	"sepX", "sepY",\
									"oriX","oriY",\
									"cohX","cohY",\
									"wallX","wallY",\
									"pvelX","pvelY","pvelT",\
									"dvelX","dvelY","dvelT"])
	elif len(sys.argv) == 3:
		model_list = cPickle.load(open(sys.argv[2]))
		best_model = find_best_model(sys.argv[1],model_list)
		outname = 'dad-lr_coeff.txt'
		print "Saving as", outname
		outf = open(outname,'w')
		for row in best_model:
			outf.write("%f %f %f\n"%(row[0],row[1],row[2]))
		outf.close()
