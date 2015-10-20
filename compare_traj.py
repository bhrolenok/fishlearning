import stats, dad, btfutil, numpy, sys
import cPickle,glob,os,os.path
import multiprocessing, multiprocessing.pool

def diff(btf1,btf2):
	btf1_traj = dad.split_btf_trajectory(btf1,['xpos','ypos','timage'],augment=False)
	btf2_traj = dad.split_btf_trajectory(btf2,['xpos','ypos','timage'],augment=False)
	errors = list()
	for eyed in btf1_traj:
		if not(eyed in btf2_traj):
			continue
		traj1 = btf1_traj[eyed]
		traj2 = btf2_traj[eyed]
		tlen = min(map(len,[traj1,traj2]))
		error = map(lambda r1,r2: numpy.linalg.norm(r1-r2), traj1[:tlen],traj2[:tlen])
		errors.append(error)
	return errors
def avg_diff(btf1,btf2):
	errors = diff(btf1,btf2)
	avg_error = list()
	avg_error_count = list()
	for e in errors:
		for eidx in range(len(e)):
			if eidx >= len(avg_error):
				avg_error.append(e[eidx])
				avg_error_count.append(1)
			else:
				avg_error[eidx]+=e[eidx]
				avg_error_count[eidx]+=1
	mean = map(lambda e,c: float(e)/float(c), avg_error, avg_error_count)
	stddev = [0,]*len(mean)
	for e in errors:
		for eidx in range(len(e)):
			stddev[eidx] += numpy.power(e[eidx]-mean[eidx],2)
	for idx in range(len(stddev)):
		stddev[idx] = numpy.sqrt(stddev[idx]/float(avg_error_count[idx]))
	return mean,stddev

def subseq_compare(subseq_pickle_filename, dad_logging_dir, output_dir, framerate=1.0/30.0):
	print "loading [",subseq_pickle_filename,"]"
	subseqs = cPickle.load(open(subseq_pickle_filename))
	print "globbing logging dirs from [",dad_logging_dir,"]"
	num_iterations = len(glob.glob(os.path.abspath(dad_logging_dir)+"/it_*_seq_0/"))
	pool = multiprocessing.pool.Pool(processes=multiprocessing.cpu_count())
	for it in range(num_iterations):
		print "Iteration",it
		print "Computing errors"
		blerp = arg_generator_gssew(dad_logging_dir,subseqs,iteration=it)
		sum_error = pool.map(get_subseq_sum_error_wrapper,blerp)
		print "Aggregating errors"
		longest_size = len(max(sum_error,key=lambda x: len(x)))
		total_error = list() #[0,]*longest_size
		total_error_count = list() #[0,]*longest_size
		for err,count in sum_error:
			for idx in range(len(err)):
				if idx >= len(total_error):
					total_error.append(err[idx])
					total_error_count.append(count[idx])
				else:
					total_error[idx] += err[idx]
					total_error_count[idx] += count[idx]
		avg_error = map(lambda x,y: float(x)/float(y),total_error,total_error_count)
		outf_name = os.path.join(output_dir,'avg_subseq_err_iter%d.dat'%it)
		print "Writting error to[",outf_name,"]"
		outf = open(outf_name,'w')
		for idx in range(len(avg_error)):
			outf.write("%f %f\n"%(float(idx*framerate),float(avg_error[idx])))
		outf.close()
		print "***"

	logs_it = glob.iglob(os.path.abspath(dad_logging_dir)+"/it_*_seq_*/")

def arg_generator_gssew(logging_dir,subseqs,iteration='*'):
	iteration = str(iteration)
	globstr = os.path.join(os.path.abspath(logging_dir),'it_'+iteration+"_*_seq_*")
	logs_it = glob.iglob(globstr)
	for log in logs_it:
		seq_num = int(log[log.rfind('_seq_')+5:])
		#print log
		sim_dir = glob.glob(os.path.join(log,'FishLRLogger-*-run-0'))[0]
		#def arggen():
		#	return {'myseq':subseqs[seq_num],'sim_logging_dir':sim_dir}
		#yield arggen
		yield {'myseq':subseqs[seq_num],'sim_logging_dir':sim_dir}

def get_subseq_sum_error_wrapper(arg_gen):
	#args = arg_gen()
	args = arg_gen
	return get_subseq_sum_error(myseq=args['myseq'],sim_logging_dir=args['sim_logging_dir'])

def get_subseq_sum_error(myseq,sim_logging_dir):
	sim_btf = btfutil.BTF()
	# sim_btf.import_from_dir(glob.glob(os.path.abspath(sim_logging_dir)+'/FishLRLogging-*-run-0'[0]))
	sim_btf.import_from_dir(sim_logging_dir)
	# tdir_name = os.path.basename(os.path.abspath(sim_logging_dir))
	# seq_num = int(tdir_name[tdir_name.rfind('_seq_')+5:])
	# mysubseq = subseqs[seq_num]
	errors = diff(myseq,sim_btf)
	sum_errors = list()
	sum_error_count = list()
	for e in errors:
		for eidx in range(len(e)):
			if eidx >= len(sum_errors):
				sum_errors.append(e[eidx])
				sum_error_count.append(1)
			else:
				sum_errors[eidx]+=e[eidx]
				sum_error_count[eidx]+=1
	return sum_errors,sum_error_count

def write_gp_data(btf1,btf2,outfname,framerate=None):
	m,s = avg_diff(btf1,btf2)
	if framerate is None:
		nextIdx = 0
		while btf1['clocktime'][0] == btf1['clocktime'][nextIdx]:
			nextIdx += 1
		framerate = float(btf1['clocktime'][nextIdx]) - float(btf1['clocktime'][0])
	outf = open(outfname,'w')
	for idx in range(len(m)):
		outf.write(str(framerate*idx)+" "+str(m[idx])+" "+str(s[idx])+"\n")
	outf.close()

if __name__ == '__main__':
	subseq_compare(sys.argv[1],sys.argv[2],sys.argv[3])
	# training_btf = btfutil.BTF()
	# training_btf.import_from_dir(sys.argv[1])
	# for idx in range(2,len(sys.argv)):
	# 	dirname = sys.argv[idx]
	# 	testing_btf = btfutil.BTF()
	# 	testing_btf.import_from_dir(dirname)
	# 	print dirname
	# 	outname = 'iter'+str(idx-1)+'.dat'
	# 	print "Iteration",idx-1,"printing to",outname
	# 	write_gp_data(training_btf,testing_btf,outname)
