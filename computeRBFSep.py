#computeRBFSep.py
import numpy
import math
import time
import sys

wait_time = 5.0 #in seconds

def compute(lines,outf,sigma):
	num_lines = len(lines)
	last_checked = time.time()
	last_line_idx = 0
	for idx in range(num_lines):
		line = map(float,lines[idx].strip().split())
		x = numpy.array(line[::2])
		y = numpy.array(line[1::2])
		weights = numpy.exp(-(numpy.power(x,2)+numpy.power(y,2))/(2.0*numpy.power(sigma,2)))
		sep_vec = numpy.sum(numpy.column_stack([x,y]).T*weights,axis=1)/float(len(x))
		outf.write(str(sep_vec[0])+" "+str(sep_vec[1])+"\n")
		cur_time = time.time()
		if (cur_time - last_checked) > wait_time:
			print "Line",idx,"lps",(idx-last_line_idx)/(cur_time-last_checked), 100*float(idx)/float(num_lines),"%"
			last_checked = cur_time
			last_line_idx = idx

if __name__ == '__main__':
	sigma = float(sys.argv[3])
	print "Sigma", sigma
	print "Writing output to",sys.argv[2]
	outf = open(sys.argv[2],'w')
	print "Reading",sys.argv[1],"...",
	sys.stdout.flush()
	lines = open(sys.argv[1]).readlines()
	print " Done!"
	compute(lines,outf,sigma)
	outf.close()