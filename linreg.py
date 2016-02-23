#linreg.py
import numpy, numpy.random, numpy.linalg, pandas
import subprocess, os, os.path
import btfutil


def generate_feature_map(n_f,D):
	# n_f = features.shape[1]
	ws = numpy.random.multivariate_normal(mean=numpy.zeros(n_f),cov=numpy.eye(n_f),size=D)
	bs = numpy.random.random(size=D)*numpy.pi*2.0
	def feature_map(feats):
		project = feats.dot(ws.T)
		rotate = numpy.tile(bs,(feats.shape[0],1))
		return numpy.sqrt(2.0/bs.shape[0])*numpy.cos(project+rotate)
	feature_map.ws = ws
	feature_map.bs = bs
	return feature_map

def learn_RFF_linreg(features,ys,cv_features=None,cv_ys=None, D=500, feature_column_names=None):
	fm = generate_feature_map(features.shape[1],D)
	rff = fm(features)
	lr_weights = numpy.linalg.lstsq(rff,ys)
	if not((cv_features is None) or (cv_ys is None)):
		print "CV error:", numpy.linalg.norm(cv_ys - (fm(cv_features).dot(lr_weights[0])))
	return lr_weights[0]

def learnLR_regularized(features,ys,cv_features=None,cv_ys=None, lamb=0.0,feature_column_names=None):
	result = numpy.linalg.lstsq(features.T.dot(features)+ lamb*numpy.identity(features.shape[1]),features.T.dot(ys))
	if not((cv_features is None) or (cv_ys is None)):
		print "CV error:",numpy.linalg.norm(cv_ys - (cv_features.dot(result[0])))
	return result[0]

def learnLR(features,ys,cv_features=None,cv_ys=None,feature_column_names=None):
	result = numpy.linalg.lstsq(features,ys)
	if not((cv_features is None) or (cv_ys is None)):
		print "CV error:",numpy.linalg.norm(cv_ys - (cv_features.dot(result[0])))
	return result[0]

def predictLR(model,num_steps,initialPlacementBTF,logdir=None):
	if logdir is None:
		logdir = os.getcwd()
	
	outname = os.path.join(logdir,'lr_coeff.txt')
	prefix = "[BTFLogger] Starting new logs in"
	outf = open(outname,'w')
	for row in model:
		outf.write("%f %f %f\n"%(row[0],row[1],row[2]))
	outf.close()
	outf = open(os.path.join(logdir,"initial_placement.txt"),"w")
	btfutil.writeInitialPlacement(outf,initialPlacementBTF)
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