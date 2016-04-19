#parseFishAndFrames.py
import sys, os.path, time
import numpy, scipy.io

VERBOSE_TIMEOUT=30

def main(matfile_fname,btf_outdir,fps):
	print "Saving BTF to dir [{}]".format(btf_outdir)
	print "Columns: {}".format(['id','ximage','yimage','timage_x','timage_y','timage','timestamp','clocktime'])
	id_btf = open(os.path.join(btf_outdir,'id.btf'),'w')
	ximage_btf = open(os.path.join(btf_outdir,'ximage.btf'),'w')
	yimage_btf = open(os.path.join(btf_outdir,'yimage.btf'),'w')
	timage_x_btf = open(os.path.join(btf_outdir,'timage_x.btf'),'w')
	timage_y_btf = open(os.path.join(btf_outdir,'timage_y.btf'),'w')
	timage_btf = open(os.path.join(btf_outdir,'timage.btf'),'w')
	timestamp_btf = open(os.path.join(btf_outdir,'timestamp.btf'),'w')
	clocktime_btf = open(os.path.join(btf_outdir,'clocktime.btf'),'w')
	print "Loading from file [{}]".format(matfile_fname)
	data = scipy.io.loadmat(matfile_fname)
	print "done."
	num_frames = len(data['frames'])
	print "Num frames:", num_frames
	print "Skipping first two frames and last frame"
	def writeit(f,d):
		f.write("{}\n".format(d))
	lasttime = time.time()
	lastidx = 2
	for frame_idx in range(2,num_frames-1):
		for detection_idx in range(len(data['frames'][frame_idx]['onfish'][0][0])):
			writeit(timestamp_btf,frame_idx)
			clocktime = float(frame_idx)/float(fps)
			writeit(clocktime_btf,clocktime)
			fishid = data['frames'][frame_idx]['onfish'][0][0][detection_idx]
			writeit(id_btf,fishid)
			ximage = data['frames'][frame_idx]['px'][0][detection_idx][0]
			writeit(ximage_btf,ximage)
			yimage = data['frames'][frame_idx]['py'][0][detection_idx][0]
			writeit(yimage_btf,yimage)
			timage_x = data['frames'][frame_idx]['vx'][0][detection_idx][0]
			writeit(timage_x_btf,timage_x)
			timage_y = data['frames'][frame_idx]['vy'][0][detection_idx][0]
			writeit(timage_y_btf,timage_y)
			timage = numpy.arctan2(timage_y,timage_x)
			writeit(timage_btf,timage)
		curtime = time.time()
		if curtime-lasttime > VERBOSE_TIMEOUT:
			print "Frame", frame_idx, "({}%)".format(100.0*float(frame_idx)/float(num_frames)),"lps", (frame_idx-lastidx)/float(VERBOSE_TIMEOUT)
			lasttime = curtime
			lastidx = frame_idx

if __name__ == '__main__':
	main(sys.argv[1],sys.argv[2],float(sys.argv[3]))
