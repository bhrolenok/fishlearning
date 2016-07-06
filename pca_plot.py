import numpy, matplotlib.pyplot, btfutil, cPickle
import pyqtgraph, pyqtgraph.opengl
app = pyqtgraph.Qt.QtGui.QApplication([])

# mw = pyqtgraph.Qt.QtGui.QMainWindow()
# view = pyqtgraph.GraphicsLayoutWidget()
# mw.setCentralWidget(view)
# mw.show()

# w1 = view.addPlot()

w = pyqtgraph.opengl.GLViewWidget()
w.resize(1800,900)
w.show()

print "Loading segments"
res = cPickle.load(open('../../fish_data/pickles/simsigma_0.15_pvel.p'))
print "Stacking data"
fnames = ['rbfsepvec', 'rbforivec','rbfcohvec', 'rbfwallvec','pvel']
feats_and_outs = map(lambda r: btfutil.btf2data(r,fnames,False),res)
all_data = numpy.row_stack([thing[0] for thing in feats_and_outs])
all_outs = numpy.row_stack([thing[1] for thing in feats_and_outs])
all_data_centered = all_data - all_data.mean(axis=0)
all_data_normalized = all_data_centered/all_data_centered.std(axis=0)

# all_data_normalized = numpy.random.rand(100,11)

print "Doing SVD"
u,s,v = numpy.linalg.svd(all_data_normalized,full_matrices=False)
#u,s,v = numpy.linalg.svd(all_data_centered,full_matrices=False)
b = v[:,:2]
down_cast = numpy.column_stack([all_data_normalized.dot(b),all_outs[:,0]])
#down_cast = all_data_centered.dot(b)
print "Plotting"
# 2D
# spi = pyqtgraph.ScatterPlotItem()
# 3D
spi = pyqtgraph.opengl.GLScatterPlotItem()
spi.setData(pos=down_cast,size=1,pxMode=True)
# w1.addItem(spi)
w.addItem(spi)
w.addItem(pyqtgraph.opengl.GLAxisItem())
# pyqtgraph.plot(x=down_cast[:,0],y=down_cast[:,1],symbol='o')
# matplotlib.pyplot.scatter(x=down_cast[:,0],y=down_cast[:,1])
# matplotlib.pyplot.show()
pyqtgraph.Qt.QtGui.QApplication.instance().exec_()
print "Done"
