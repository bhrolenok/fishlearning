#computePrevVel.py
import numpy
import math
import time
import sys
import btfutil


wait_time = 5.0 #in seconds

def compute(btf,outf,spf):
	num_lines = len(btf['id'])
	prev_block_start = 0
	cur_block_start = None
	for idx in range(num_lines):
		if btf['timestamp'][idx] != btf['timestamp'][prev_block_start]:
			cur_block_start = idx
			break
		outf.write("0.0 0.0 0.0\n")
	if cur_block_start is None:
		raise RuntimeError("Couldn't find end of first block")
	last_checked = time.time()
	last_line_idx = 0
	for idx in range(cur_block_start,num_lines):
		if btf['timestamp'][cur_block_start] != btf['timestamp'][idx]:
			prev_block_start = cur_block_start
			cur_block_start = idx
		id_found = False
		xvel,yvel,tvel = 0.0,0.0,0.0
		for bidx in range(prev_block_start,cur_block_start):
			if btf['id'][idx] == btf['id'][bidx]:
				xvel = (float(btf['xpos'][idx]) - float(btf['xpos'][bidx]))/spf
				yvel = (float(btf['ypos'][idx]) - float(btf['ypos'][bidx]))/spf
				tvel = float(btf['timage'][idx]) - float(btf['timage'][bidx])
				if tvel > math.pi:
					tvel = tvel - 2*math.pi
				elif tvel < -math.pi:
					tvel = tvel + 2*math.pi
				tvel = tvel/spf
				break
		outf.write("{} {} {}\n".format(xvel,yvel,tvel))
		cur_time = time.time()
		if (cur_time - last_checked) > wait_time:
			print "Line",idx,"lps",(idx-last_line_idx)/(cur_time-last_checked), 100*float(idx)/float(num_lines),"%"
			last_checked = cur_time
			last_line_idx = idx

if __name__ == '__main__':
	print "Writing output to",sys.argv[2]
	outf = open(sys.argv[2],'w')
	print "Loading btf's from",sys.argv[1]
	btf = btfutil.BTF()
	btf.import_from_dir(sys.argv[1])
	print "Using", sys.argv[3], "seconds per frame"
	compute(btf,outf,float(sys.argv[3]))
	outf.close()