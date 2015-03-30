import btfutil, numpy, sys
import matplotlib.pyplot as plt

def maxDist(data):
	mx = None
	for eye in range(len(data)):
		for jay in range(eye+1,len(data)):
			# print data[eye,:],data[jay,:]
			tmp = numpy.linalg.norm(data[eye,:]-data[jay,:])
			if mx is None or tmp > mx:
				mx = tmp
	return mx

def nnDists(data):
	dists = list()
	for eye in range(len(data)):
		mn = None
		for jay in range(len(data)):
			if jay == eye:
				continue
			else:
				tmp = numpy.linalg.norm(data[eye,:]-data[jay,:])
				if mn is None or tmp < mn:
					mn = tmp
		dists.append(mn)
	return numpy.array(dists)

def avgNNDist(data):
	return nnDists(data).mean()

def varNNDist(data):
	return nnDists(data).std()

def write_gnuplot(outfname,x,y):
	outf = open(outfname,"w")
	for rowIdx in range(len(x)):
		outf.write(str(x[rowIdx])+" "+str(y[rowIdx])+"\n")
	outf.close()

if __name__ == '__main__':
	btf_gen = btfutil.BTF()
	btf_gen.import_from_dir(sys.argv[1])
	btf_gen.filter_by_col('dbool')
	btf_learned = btfutil.BTF()
	btf_learned.import_from_dir(sys.argv[2])
	btf_learned.filter_by_col('dbool')
	#max
	gen_max = btfutil.timeseries(btf_gen,maxDist,['xpos','ypos','timage'])
	write_gnuplot(sys.argv[3]+"_gen_timeseries_max.dat",gen_max[0],gen_max[1])
	learned_max = btfutil.timeseries(btf_learned,maxDist,['xpos','ypos','timage'])
	write_gnuplot(sys.argv[3]+"_learned_timeseries_max.dat",learned_max[0],learned_max[1])
	#avg
	gen_avg = btfutil.timeseries(btf_gen,avgNNDist,['xpos','ypos','timage'])
	write_gnuplot(sys.argv[3]+"_gen_timeseries_avg.dat",gen_avg[0],gen_avg[1])
	learned_avg = btfutil.timeseries(btf_learned,avgNNDist,['xpos','ypos','timage'])
	write_gnuplot(sys.argv[3]+"_learned_timeseries_avg.dat",learned_avg[0],learned_avg[1])
	#std
	gen_std = btfutil.timeseries(btf_gen,varNNDist,['xpos','ypos','timage'])
	write_gnuplot(sys.argv[3]+"_gen_timeseries_std.dat",gen_std[0],gen_std[1])
	learned_std = btfutil.timeseries(btf_learned,varNNDist,['xpos','ypos','timage'])
	write_gnuplot(sys.argv[3]+"_learned_timeseries_std.dat",learned_std[0],learned_std[1])