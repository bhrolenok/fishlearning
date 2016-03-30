#fixDvel.py
import numpy
import math
import time
import sys
import btfutil


wait_time = 5.0 #in seconds

def compute(btf,outf,spf):
	num_lines = len(btf['id'])
	last_checked = time.time()
	last_line_idx = 0
	num_lines_changed = 0
	for idx in range(num_lines):
		xvel,yvel,tvel = map(float,btf['dvel'][idx].split())
		if tvel*spf > math.pi:
			tvel = tvel*spf
			tvel = tvel-2.0*math.pi
			tvel = tvel/spf
			num_lines_changed += 1
		elif tvel*spf < -math.pi:
			tvel = tvel*spf
			tvel = tvel+2.0*math.pi
			tvel = tvel/spf			
			num_lines_changed += 1
		outf.write("{} {} {}\n".format(xvel,yvel,tvel))
		cur_time = time.time()
		if (cur_time - last_checked) > wait_time:
			print "Line",idx,"lps",(idx-last_line_idx)/(cur_time-last_checked), 100*float(idx)/float(num_lines),"%"
			last_checked = cur_time
			last_line_idx = idx
	return num_lines_changed

if __name__ == '__main__':
	print "Writing output to",sys.argv[2]
	outf = open(sys.argv[2],'w')
	print "Loading btf's from",sys.argv[1]
	btf = btfutil.BTF()
	btf.import_from_dir(sys.argv[1])
	print "Using", sys.argv[3], "seconds per frame"
	nlf = compute(btf,outf,float(sys.argv[3]))
	print "Fixed",nlf,"lines"
	outf.close()