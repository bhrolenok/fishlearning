import numpy, btfutil, scipy.spatial

class KNN():
	def __init__(self,features,ys):
		self.kdt = scipy.spatial.cKDTree(features)
		self.ys = ys
	def query(self,features,k):
		self.ys[self.kdt.query(features,k)[1]]

def learnLR(features,ys):
	return numpy.linalg.lstsq(features,ys)[0]
def learnKNN(features,ys):
	return KNN(features,ys)

def btf2data(btf,feature_names):
	features = numpy.column_stack([map(lambda line:: map(float,line.split()), btf[col_name]) for col_name in feature_names])
	ys = numpy.array(map(lambda line: map(float, line.split()), btf['dvel']))
	filt = numpy.array(btf['dbool']) == 'True'
	return features[filt],ys[filt]

def split_btf_trajectory(btf):
	features,ys = btf2data(btf,['xpos','ypos','timage'])
	unique_ids = set(btf['id'])
	npid = numpy.array(btf['id'])
	return {eyed:features[npid==eyed] for eyed in unique_ids}

def dad(N,btf,learn,predict,feature_names = ['rbfsepvec','rbforivec','rbfcohvec','rbfwallvec']):
	training_features, training_ys = btf2data(btf,feature_names)
	models = (learn(training_features,training_ys),)
	for n in range(N):
		sim_btf = predict(models[n])
		sim_features,sim_ys = btf2data(sim_btf,feature_names)