#knn.py
import numpy, scipy.spatial, pandas
import tarfile, shutil, os, os.path, bz2
import subprocess
import btfutil
import sys, traceback

class KNN():
	def __init__(self,features,ys):
		self.kdt = scipy.spatial.cKDTree(features)
		self.ys = ys
	def query(self,features,k):
		self.ys[self.kdt.query(features,k)[1]]
	def to_csv(self,outf,feature_names):
		data_df = pandas.DataFrame(numpy.column_stack(self.kdt.data,self.ys),columns=feature_names)
		data_df.to_csv(outf,index=False)

def predictKNN_singleAgent(model, num_steps, initialPlacementBTF, logdir=None):
	if logdir is None:
		logdir = os.getcwd()
	outname = os.path.join(logdir,'knn_dataset.csv')
	exampleBTFDir = os.path.join(logdir,'exampleBTF')
	os.mkdir(exampleBTFDir)
	prefix = "[BTFLogger] Starting new logs in"
	outf = open(outname,'w')
	model.to_csv(outf,index=False)
	outf.close()
	placementFname = os.path.join(logdir,'initial_placement.txt')
	outf = open(placementFname,'w')
	btfutil.writeInitialPlacement(outf,initialPlacementBTF)
	outf.close()
	firstID = initialPlacementBTF['id'][0]
	initialPlacementBTF.save_to_dir(exampleBTFDir)
	proc = subprocess.Popen(['java',\
							'biosim.app.fishreynolds.FishReynolds',\
							'-placed',placementFname,\
							'-nogui',\
							'-logging',logdir,\
							'-knn', outname,\
							'-replay',exampleBTFDir,\
							'-ignoreTrackIDs',firstID,\
							'-for',str(num_steps)],\
							stdout = subprocess.PIPE, stderr=subprocess.PIPE)
	output,errors = proc.communicate()
	if proc.returncode != 0:
		raise Exception("[knn.py] Error executing simmulation:\n Output:\n{}\n Errors:\n{}".format(output,errors))
	#print "output:\n",output
	#print "errors:\n",errors
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
	return [rv,]

def predictKNN_allAgents(model, num_steps, initialPlacementBTF,logdir=None):
	# if logdir is None:
	# 	logdir = os.getcwd()
	# outname = os.path.join(logdir,'knn_dataset.csv')
	# prefix = "[BTFLogger] Starting new logs in"
	# outf = open(outname,'w')
	# model.to_csv(outf,index=False)
	# outf.close()
	# placementFname = os.path.join(logdir,'initial_placement.txt')
	# outf = open(placementFname,'w')
	# btfutil.writeInitialPlacement(outf,initialPlacementBTF)
	# outf.close()
	# proc = subprocess.Popen(['java',\
	# 						'biosim.app.fishreynolds.FishReynolds',\
	# 						'-placed',placementFname,\
	# 						'-nogui',\
	# 						'-logging',logdir,\
	# 						'-knn', outname,\
	# 						'-for',str(num_steps)],\
	# 						stdout = subprocess.PIPE, stderr=subprocess.PIPE)
	# output,errors = proc.communicate()
	# trace_btfdir_start = len(prefix)+output.index(prefix)
	# trace_btfdir_end = output.index('\n',trace_btfdir_start)
	# trace_btfdir = output[trace_btfdir_start:trace_btfdir_end].strip()
	# tf = tarfile.open(logdir+".tar.bz2",mode='w:bz2')
	# tf.add(logdir)
	# tf.close()
	# shutil.rmtree(logdir)
	# rv = btfutil.BTF()
	# #rv.import_from_dir(trace_btfdir)
	# rv.import_from_tar(logdir+".tar.bz2")
	# rv.filter_by_col('dbool')
	# return list(rv)
	if logdir is None:
		logdir = os.getcwd()
	outname = os.path.join(logdir,'knn_dataset.csv')
	exampleBTFDir = os.path.join(logdir,'exampleBTF')
	os.mkdir(exampleBTFDir)
	prefix = "[BTFLogger] Starting new logs in"
	outf = open(outname,'w')
	model.to_csv(outf,index=False)
	outf.close()
	placementFname = os.path.join(logdir,'initial_placement.txt')
	outf = open(placementFname,'w')
	numInitIds = btfutil.writeInitialPlacement(outf,initialPlacementBTF)
	outf.close()
	firstID = initialPlacementBTF['id'][0]
	initialPlacementBTF.save_to_dir(exampleBTFDir)
	rv = list()
	for activeID in range(numInitIds):
		proc = subprocess.Popen(['java',\
								'biosim.app.fishreynolds.FishReynolds',\
								'-placed',placementFname,\
								'-nogui',\
								'-logging',logdir,\
								'-knn', outname,\
								'-replay',exampleBTFDir,\
								'-ignoreTrackIDs',initialPlacementBTF['id'][activeID],\
								'-for',str(num_steps)],\
								stdout = subprocess.PIPE, stderr=subprocess.PIPE)
		output,errors = proc.communicate()
		if proc.returncode != 0:
			raise Exception("[knn.py] Error executing simmulation:\n Output:\n{}\n Errors:\n{}".format(output,errors))
		#print "output:\n",output
		#print "errors:\n",errors
		trace_btfdir_start = len(prefix)+output.index(prefix)
		trace_btfdir_end = output.index('\n',trace_btfdir_start)
		trace_btfdir = output[trace_btfdir_start:trace_btfdir_end].strip()
		tf_name = trace_btfdir+".tar.bz2"
		tf = tarfile.open(tf_name,mode='w:bz2')
		tf.add(trace_btfdir)
		tf.close()
		shutil.rmtree(trace_btfdir)
		tmprv = btfutil.BTF()
		#tmprv.import_from_dir(trace_btfdir)
		tmprv.import_from_tar(tf_name)
		#tmprv.filter_by_col('dbool')
		#tmprv.load_all_columns()
		#print tmprv['id']
		rv.append(tmprv)
	#tf = tarfile.open(logdir+".tar.bz2",mode='w:bz2')
	bzf = bz2.BZ2File(outname+".bz2",mode='w')
	bzf.writelines((open(outname)).readlines())
	bzf.close()
	os.remove(outname)
	#tf.add(logdir)
	#tf.close()
	#shutil.rmtree(logdir)
	return rv

def predictKNN(model, num_steps, initialPlacementBTF,logdir=None):
	# return predictKNN_singleAgent(model,num_steps,initialPlacementBTF,logdir)
	return predictKNN_allAgents(model,num_steps,initialPlacementBTF,logdir)

def learnKNN(features,ys,cv_features=None,cv_ys=None, feature_column_names=None):
	if not(cv_features is None):
		knn_kdt = scipy.spatial.cKDTree(features) #KNN(features=features,ys=ys)
		knn_ys = ys[knn_kdt.query(cv_features,3)[1]].mean(axis=1)
		print "CV error:",numpy.linalg.norm(cv_ys - knn_ys)
	combined = numpy.column_stack((features,ys),)
	#print combined.shape, feature_column_names
	return pandas.DataFrame(combined,columns=feature_column_names)
