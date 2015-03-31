import numpy, btfutil, scipy.spatial, subprocess, time, sys, cPickle

class KNN():
	def __init__(self,features,ys):
		self.kdt = scipy.spatial.cKDTree(features)
		self.ys = ys
	def query(self,features,k):
		self.ys[self.kdt.query(features,k)[1]]

def predictLR(model,num_steps,initialPlacementBTFDir):
	outname = 'lr_coeff.txt'
	prefix = "[BTFLogger] Starting new logs in"
	outf = open(outname,'w')
	for row in model:
		outf.write("%f %f %f\n"%(row[0],row[1],row[2]))
	outf.close()
	proc = subprocess.Popen(['java','biosim.app.fishlr.FishLR','-placed','-btf',initialPlacementBTFDir,'-nogui','-logging', '-lr', outname,'-for',str(num_steps)],stdout=subprocess.PIPE)
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


def main(training_dir,num_models,max_seq_len):
	models = dad(num_models,max_seq_len,training_dir,learnLR,predictLR)
	# print models
	cPickle.dump(models,open("dad-results.p","w"))

if __name__ == '__main__':
	if len(sys.argv) == 4:
		main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
	elif len(sys.argv) == 3:
		model_list = cPickle.load(open(sys.argv[2]))
		best_model = find_best_model(sys.argv[1],model_list)
		outname = 'dad-lr_coeff.txt'
		print "Saving as", outname
		outf = open(outname,'w')
		for row in best_model:
			outf.write("%f %f %f\n"%(row[0],row[1],row[2]))
		outf.close()
