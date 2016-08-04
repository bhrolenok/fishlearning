import numpy, matplotlib.pyplot, btfutil, cPickle, scipy.stats
import pyqtgraph, pyqtgraph.opengl
import sys

# mw = pyqtgraph.Qt.QtGui.QMainWindow()
# view = pyqtgraph.GraphicsLayoutWidget()
# mw.setCentralWidget(view)
# mw.show()

# w1 = view.addPlot()

# w = pyqtgraph.opengl.GLViewWidget()
# w.resize(1800,900)
# w.show()


# seg_file_path = '../../fish_data/pickles/simsigma_0.15_pvel.p'
# sim_file_path = '../../Desktop/FishLRLogger-knnsamplenorm-july-07-2016-13.26/'
# sim_file_path = '../../Desktop/FishLRLogger-knnaverage-july-07-2016-14.04/'
# sim_file_path = '../../Desktop/FishLRLogger-linreg-july-07-2016-14.40/'
real_file_path = '../../fish_data/java_btf_seqs/'
if len(sys.argv) == 2:
	real_file_path = sys.argv[1]

print "Loading segments"
res = btfutil.load_sequence_dir(real_file_path)

print "Stacking data"
# fnames = ['rbfsepvec', 'rbforivec','rbfcohvec', 'rbfwallvec','pvel']
fnames = ['pvel']
feats_and_outs = map(lambda r: btfutil.btf2data(r,fnames,False),res)
all_outs = numpy.row_stack([thing[1] for thing in feats_and_outs])
all_feats = numpy.row_stack([thing[0] for thing in feats_and_outs])

print "Computing delta-vel"
delta_vel = all_outs-all_feats

print "Creating histograms"
xvel_y, xvel_x = numpy.histogram(all_outs[:,0],bins=150,range=(-0.5,0.5),density=True)
# yvel_y, yvel_x = numpy.histogram(delta_vel[:,1],bins=150,range=(-0.5,0.5),density=True)
# tvel_y, tvel_x = numpy.histogram(delta_vel[:,2],bins=150,range=(-0.5,0.5),density=True)
dxvel_y, dxvel_x = numpy.histogram(delta_vel[:,0],bins=150,range=(-0.5,0.5),density=True)

print "Plotting"
app = pyqtgraph.Qt.QtGui.QApplication([])
#xVel
w1 = pyqtgraph.GraphicsWindow(title="xvel")
plt1 = w1.addPlot()
plt1.plot(xvel_x, xvel_y,stepMode=True, fillLevel=0,brush=(0,0,255,150))
xvel_mean = all_outs[:,0].mean()
xvel_std = all_outs[:,0].std()
print "xvel mean +- std:", xvel_mean,"+-",xvel_std
plt1.addItem(pyqtgraph.InfiniteLine(xvel_mean,pen={'style':1,'color':(255,255,255,150)}))
plt1.addItem(pyqtgraph.InfiniteLine(xvel_mean+xvel_std,pen={'style':3,'color':(255,255,255,150)}))
plt1.addItem(pyqtgraph.InfiniteLine(xvel_mean-xvel_std,pen={'style':3,'color':(255,255,255,150)}))
plt1.addItem(pyqtgraph.TextItem("{} +- {}".format(xvel_mean,xvel_std)))

#xVel
w2 = pyqtgraph.GraphicsWindow(title="delta xvel")
plt2 = w2.addPlot()
plt2.plot(dxvel_x, dxvel_y,stepMode=True, fillLevel=0,brush=(0,0,255,150))
dxvel_mean = delta_vel[:,0].mean()
dxvel_std = delta_vel[:,0].std()
print "dxvel mean +- std:", dxvel_mean,"+-",dxvel_std
plt2.addItem(pyqtgraph.InfiniteLine(dxvel_mean,pen={'style':1,'color':(255,255,255,150)}))
plt2.addItem(pyqtgraph.InfiniteLine(dxvel_mean+xvel_std,pen={'style':3,'color':(255,255,255,150)}))
plt2.addItem(pyqtgraph.InfiniteLine(dxvel_mean-xvel_std,pen={'style':3,'color':(255,255,255,150)}))
plt2.addItem(pyqtgraph.TextItem("{} +- {}".format(dxvel_mean,dxvel_std)))


# combo xVel
w_combo = pyqtgraph.GraphicsWindow(title="Combined xVel - delta-xVel")
plt_c = w_combo.addPlot()
plt_c.plot(xvel_x, xvel_y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
plt_c.addItem(pyqtgraph.InfiniteLine(xvel_mean,pen={'style':1,'color':(255,255,255,150)}))
plt_c.addItem(pyqtgraph.InfiniteLine(xvel_mean+xvel_std,pen={'style':3,'color':(255,255,255,150)}))
plt_c.addItem(pyqtgraph.InfiniteLine(xvel_mean-xvel_std,pen={'style':3,'color':(255,255,255,150)}))
plt_c.plot(dxvel_x, dxvel_y,stepMode=True,fillLevel=0,brush=(0,255,0,150))
plt_c.addItem(pyqtgraph.InfiniteLine(dxvel_mean,pen={'style':1,'color':(0,255,0,150)}))
plt_c.addItem(pyqtgraph.InfiniteLine(dxvel_mean+dxvel_std,pen={'style':3,'color':(0,255,0,150)}))
plt_c.addItem(pyqtgraph.InfiniteLine(dxvel_mean-dxvel_std,pen={'style':3,'color':(0,255,0,150)}))
# plt_c.addItem(pyqtgraph.TextItem("pval: {}, stat: {}".format(cstest_res[1],cstest_res[0])))

pyqtgraph.Qt.QtGui.QApplication.instance().exec_()
print "Done"
