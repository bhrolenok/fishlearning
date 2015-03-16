import numpy, btfutil, scipy.spatial, subprocess, time

class KNN():
	def __init__(self,features,ys):
		self.kdt = scipy.spatial.cKDTree(features)
		self.ys = ys
	def query(self,features,k):
		self.ys[self.kdt.query(features,k)[1]]

def predictLR(model,num_steps):
	outname = 'lr_coeff.txt'
	prefix = "[BTFLogger] Starting new logs in"
	outf = open(outname,'w')
	for row in model:
		outf.write("%f %f %f\n"%(row[0],row[1],row[2]))
	outf.close()
	proc = subprocess.Popen(['java','biosim.app.fishlr.FishLR','-nogui','-logging','-for',str(num_steps)],stdout=subprocess.PIPE)
	output,errors = proc.communicate()
	trace_btfdir_start = len(prefix)+output.index(prefix)
	trace_btfdir_end = output.index("\n",trace_btfdir_start)
	trace_btfdir = output[trace_btfdir_start:trace_btfdir_end].strip()
	rv = btfutil.BTF()
	rv.import_from_dir(trace_btfdir)
	rv.filter_by_col('dbool')
	return rv

def learnLR(features,ys):
	result = numpy.linalg.lstsq(features,ys)
	print "Residuals:",result[1]
	return result[0]
def learnKNN(features,ys):
	return KNN(features,ys)

def btf2data(btf,feature_names):
	features = numpy.column_stack([map(lambda line: map(float,line.split()), btf[col_name]) for col_name in feature_names])
	ys = numpy.array(map(lambda line: map(float, line.split()), btf['dvel']))
	return features,ys

def split_btf_trajectory(btf,feature_names):
	features,ys = btf2data(btf,feature_names)
	npid = numpy.array(map(int,btf['id']))
	unique_ids = set(npid)
	return {eyed:features[npid==eyed] for eyed in unique_ids}

def dad(N,k,btf,learn,predict,feature_names = ['rbfsepvec','rbforivec','rbfcohvec','rbfwallvec']):
	btf.filter_by_col('dbool')
	training_features, training_ys = btf2data(btf,feature_names)
	training_trajectory = split_btf_trajectory(btf,['xpos','ypos','timage'])
	predict_steps = min(map(lambda key: training_trajectory[key].shape[0], training_trajectory),k)
	training_traj_features = split_btf_trajectory(btf,feature_names)
	models = (learn(training_features,training_ys),)
	for n in range(N):
		sim_btf = predict(models[n],predict_steps)
		sim_features,sim_ys = btf2data(sim_btf,feature_names)
		sim_trajectory = split_btf_trajectory(sim_btf,['xpos','ypos','timage'])
		sim_traj_features = split_btf_trajectory(sim_btf,feature_names)
		dad_training_features, dad_training_ys = training_features, training_ys
		for eyed in sim_trajectory:
			traj = sim_trajectory[eyed]
			traj_feats = sim_traj_features[eyed]
			for row_idx in range(1,min(training_trajectory[eyed+1].shape[0]-1,tran.shape[0]-1)):
				dad_sample_feats = traj_feats[row_idx]
				dad_sample_ys = training_trajectory[eyed-1][row_idx+1]-traj[row_idx]
				dad_training_features = dad_training_features.append(dad_sample_feats)
				dad_training_ys = dad_training_ys.append(dad_sample_ys)
		models = models + (learn(dad_training_features,dad_training_ys),)
	return models

def debug():
	training_btf = btfutil.BTF()
	training_btf.import_from_dir('data/')
	models = dad(10,10,training_btf,learnLR,predictLR)
	print models

if __name__ == '__main__':
	debug()