import glob, os.path, string, numpy, time

VERBOSE_TIMEOUT=30 #Time in seconds between print outs

def verbose_readlines(infile):
	theline = infile.readline()
	lasttime = time.time()
	lastidx = 0
	curidx = 0
	while theline != "":
		yield theline
		theline = infile.readline()
		curtime = time.time()
		if curtime - lasttime > VERBOSE_TIMEOUT:
			if lastidx == 0:
				print "[BTFUtil] Loading",infile.name
			print "[BTFUtil] Line", curidx, "lps",(curidx-lastidx)/float(VERBOSE_TIMEOUT)
			lastidx = curidx
			lasttime = curtime
		curidx += 1

class BTF:
	def __init__(self):
		self.column_filenames = dict()
		self.column_data = dict()
		self.mask = None

	def import_from_dir(self,dirname):
		self.mask = None
		new_columns = glob.glob(os.path.join(dirname,'*.btf'))
		for column in new_columns:
			cname = os.path.basename(column)[:-4]
			self.column_filenames[cname] = column
	def load_column(self,cname):
		if cname in self.column_filenames:
			self.column_data[cname] = tuple(map(string.strip, verbose_readlines(open(self.column_filenames[cname]))))
			return True
		return False

	def load_columns(self,cnames):
		return {k: self.load_column(k) for k in cnames}

	def load_all_columns(self):
		return self.load_columns(self.column_filenames.keys())

	def has_columns(self,cnames):
		for c in cnames:
			if not(c in self):
				return False
		return True

	def filter_by_col(self,col):
		self.mask = [ele.capitalize() == 'True' for ele in self[col]]
		pass

	def to_nparr(self):
		for key in self.column_data:
			col_name = key
			col_data = numpy.array(map(lambda s: s.split(), self[key]))
			print col_data
			col_width = col_data.shape[1]
		rv = numpy.column_stack([self.column_data[key] for key in self.column_data])
		rv = numpy.array(rv,dtype=[(key,'float') for key in self.column_data])
		return rv

	def __contains__(self,key):
		return key in self.column_filenames

	def __getitem__(self,key):
		if not(self.__contains__(key)):
			raise KeyError("No column named ["+str(key)+"]")
		rv = None
		if key in self.column_data:
			rv = self.column_data[key]
		else:
			if self.load_column(key):
				rv = self.column_data[key]
			else:
				raise KeyError("Could not load BTF column: ("+str(key)+","+str(self.column_filenames[key])+")")
		if self.mask is None:
			return rv
		else:
			return filter(lambda d: not(d is None), map(lambda data,mask: data if mask else None, rv, self.mask))

def timeseries(btf,fun,pColNames,tCol='clocktime'):
	oldT = None
	firstNewT = 0
	alldata = list()
	alltimes = list()
	for tIdx in range(len(btf[tCol])):
		if (oldT is None):
			oldT = btf[tCol][tIdx]
		elif btf[tCol][tIdx] != oldT:
			cdata = numpy.column_stack([map(float,btf[pName][firstNewT:tIdx]) for pName in pColNames])
			alldata.append(fun(cdata))
			alltimes.append(float(oldT))
			firstNewT = tIdx
			oldT = btf[tCol][tIdx]
	return numpy.array(alltimes), numpy.array(alldata)

def printif(s,q):
	if q:
		print s

def split_subsequences(btf,subseq_length_t,ignore_shorter=True,depth=0,debug=False):
	printif("depth %d"%depth,debug)
	head_btf = BTF()
	tail_btf = BTF()
	head_btf.column_filenames = btf.column_filenames
	tail_btf.column_filenames = btf.column_filenames
	seq_start_t = float(btf['clocktime'][0])
	block_start_idx = 0
	id_set = None
	max_len = len(btf['clocktime'])
	if ((float(btf['clocktime'][max_len-1])-seq_start_t)<subseq_length_t) and (ignore_shorter):
		printif("Final segment too short",debug)
		return tuple()
	while block_start_idx < max_len and (float(btf['clocktime'][block_start_idx])-seq_start_t)<subseq_length_t:
		block_end_idx=block_start_idx
		tmp_id_set = set()
		while block_end_idx < max_len and float(btf['clocktime'][block_end_idx])==float(btf['clocktime'][block_start_idx]):
			tmp_id_set.add(btf['id'][block_end_idx])
			block_end_idx += 1
		if id_set is None:
			id_set = tmp_id_set
		block_start_idx = block_end_idx
		if id_set != tmp_id_set:
			break
	last_seq_idx = min(max_len-1,block_start_idx)
	printif("seq length %fs"%(float(btf['clocktime'][last_seq_idx])-seq_start_t),debug)
	printif("last_seq_idx %d"%last_seq_idx,debug)
	for key in btf.column_filenames:
		if not(key in btf.column_data):
			btf.load_column(key)
		tail_btf.column_data[key] = btf.column_data[key][last_seq_idx:]
		head_btf.column_data[key] = btf.column_data[key][:last_seq_idx]
		if not(btf.mask is None):
			tail_btf.mask = btf.mask[last_seq_idx:]
			head_btf.mask = btf.mask[:last_seq_idx]
	if (float(btf['clocktime'][last_seq_idx])-seq_start_t)<subseq_length_t and ignore_shorter:
		printif("ended early",debug)
		rv = tuple()
	else:
		rv = (head_btf,)
	return rv+split_subsequences(tail_btf,subseq_length_t,ignore_shorter,depth+1,debug)