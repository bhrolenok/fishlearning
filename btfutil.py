import glob, os.path, string

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
			self.column_data[cname] = map(string.strip, open(self.column_filenames[cname]).readlines())
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