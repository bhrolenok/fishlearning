import stats, dad, btfutil, numpy

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