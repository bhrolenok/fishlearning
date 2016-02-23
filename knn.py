#knn.py
import numpy, scipy.spatial, pandas
import tarfile, shutil, os, os.path
import subprocess
import btfutil

class KNN():
	def __init__(self,features,ys):
		self.kdt = scipy.spatial.cKDTree(features)
		self.ys = ys
	def query(self,features,k):
		self.ys[self.kdt.query(features,k)[1]]
	def to_csv(self,outf,feature_names):
		data_df = pandas.DataFrame(numpy.column_stack(self.kdt.data,self.ys),columns=feature_names)
		data_df.to_csv(outf,index=False)

def predictKNN(model, num_steps, initialPlacementBTF,logdir=None):
	if logdir is None:
		logdir = os.getcwd()
	outname = os.path.join(logdir,'knn_dataset.csv')
	prefix = "[BTFLogger] Starting new logs in"
	outf = open(outname,'w')
	model.to_csv(outf,index=False)
	outf.close()
	placementFname = os.path.join(logdir,'initial_placement.txt')
	outf = open(placementFname,'w')
	btfutil.writeInitialPlacement(outf,initialPlacementBTF)
	outf.close()
	proc = subprocess.Popen(['java',\
							'biosim.app.fishreynolds.FishReynolds',\
							'-placed',placementFname,\
							'-nogui',\
							'-logging',logdir,\
							'-knn', outname,\
							'-for',str(num_steps)],\
							stdout = subprocess.PIPE, stderr=subprocess.PIPE)
	output,errors = proc.communicate()
	trace_btfdir_start = len(prefix)+output.index(prefix)
	trace_btfdir_end = output.index('\n',trace_btfdir_start)
	trace_btfdir = output[trace_btfdir_start:trace_btfdir_end].strip()
	tf = tarfile.open(logdir+".tar.bz2",mode='w:bz2')
	tf.add(logdir)
	tf.close()
	shutil.rmtree(logdir)
	rv = btfutil.BTF()
	#rv.import_from_dir(trace_btfdir)
	rv.import_from_tar(logdir+".tar.bz2")
	rv.filter_by_col('dbool')
	return rv

def learnKNN(features,ys,cv_features=None,cv_ys=None, feature_column_names=None):
	if not(cv_features is None):
		knn_kdt = scipy.spatial.cKDTree(features) #KNN(features=features,ys=ys)
		knn_ys = ys[knn_kdt.query(cv_features,3)[1]].mean(axis=1)
		print "CV error:",numpy.linalg.norm(cv_ys - knn_ys)
	combined = numpy.column_stack((features,ys),)
	return pandas.DataFrame(combined,columns=feature_column_names)
