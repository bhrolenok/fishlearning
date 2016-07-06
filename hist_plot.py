import numpy, matplotlib.pyplot, btfutil, cPickle
import pyqtgraph, pyqtgraph.opengl

# mw = pyqtgraph.Qt.QtGui.QMainWindow()
# view = pyqtgraph.GraphicsLayoutWidget()
# mw.setCentralWidget(view)
# mw.show()

# w1 = view.addPlot()

# w = pyqtgraph.opengl.GLViewWidget()
# w.resize(1800,900)
# w.show()

print "Loading segments"
seg_file_path = '../../fish_data/pickles/simsigma_0.15_pvel.p'
res = cPickle.load(open(seg_file_path))
sim_res = btfutil.BTF()
sim_file_path = '../../Desktop/FishLRLogger-knn-it0/'
sim_res.import_from_dir(sim_file_path)
print "Stacking data"
fnames = ['rbfsepvec', 'rbforivec','rbfcohvec', 'rbfwallvec','pvel']
feats_and_outs = map(lambda r: btfutil.btf2data(r,fnames,False),res)
# all_data = numpy.row_stack([thing[0] for thing in feats_and_outs])
all_outs = numpy.row_stack([thing[1] for thing in feats_and_outs])
# all_data_centered = all_data - all_data.mean(axis=0)
# all_data_normalized = all_data_centered/all_data_centered.std(axis=0)
sim_outs = btfutil.btf2data(sim_res,fnames,False)[1]
# all_outs = numpy.random.rand(100,11)

print "Creating histograms"
xvel_y, xvel_x = numpy.histogram(all_outs[:,0],bins=150)
yvel_y, yvel_x = numpy.histogram(all_outs[:,1],bins=150)
tvel_y, tvel_x = numpy.histogram(all_outs[:,2],bins=150)
sim_xvel_y, sim_xvel_x = numpy.histogram(sim_outs[:,0],bins=150)
sim_yvel_y, sim_yvel_x = numpy.histogram(sim_outs[:,1],bins=150)
sim_tvel_y, sim_tvel_x = numpy.histogram(sim_outs[:,2],bins=150)

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

#sim xVel
w_sim = pyqtgraph.GraphicsWindow(title="Sim xVel")
plt_sim = w_sim.addPlot()
plt_sim.plot(sim_xvel_x, sim_xvel_y,stepMode=True,fillLevel=0,brush=(0,255,0,150))
sim_xvel_mean = sim_outs[:,0].mean()
sim_xvel_std = sim_outs[:,0].std()
print "sim xvel mean +- std:", sim_xvel_mean,"+-",sim_xvel_std
plt_sim.addItem(pyqtgraph.InfiniteLine(sim_xvel_mean,pen={'style':1,'color':(0,255,0,150)}))
plt_sim.addItem(pyqtgraph.InfiniteLine(sim_xvel_mean+sim_xvel_std,pen={'style':3,'color':(0,255,0,150)}))
plt_sim.addItem(pyqtgraph.InfiniteLine(sim_xvel_mean-sim_xvel_std,pen={'style':3,'color':(0,255,0,150)}))
plt_sim.addItem(pyqtgraph.TextItem("{} +- {}".format(sim_xvel_mean,sim_xvel_std)))

#combo xVel
w_combo = pyqtgraph.GraphicsWindow(title="Combined xVel")
plt_c = w_combo.addPlot()
plt_c.plot(xvel_x, xvel_y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
plt_c.addItem(pyqtgraph.InfiniteLine(xvel_mean,pen={'style':1,'color':(255,255,255,150)}))
plt_c.addItem(pyqtgraph.InfiniteLine(xvel_mean+xvel_std,pen={'style':3,'color':(255,255,255,150)}))
plt_c.addItem(pyqtgraph.InfiniteLine(xvel_mean-xvel_std,pen={'style':3,'color':(255,255,255,150)}))
plt_c.plot(sim_xvel_x, sim_xvel_y,stepMode=True,fillLevel=0,brush=(0,255,0,150))
plt_c.addItem(pyqtgraph.InfiniteLine(sim_xvel_mean,pen={'style':1,'color':(0,255,0,150)}))
plt_c.addItem(pyqtgraph.InfiniteLine(sim_xvel_mean+sim_xvel_std,pen={'style':3,'color':(0,255,0,150)}))
plt_c.addItem(pyqtgraph.InfiniteLine(sim_xvel_mean-sim_xvel_std,pen={'style':3,'color':(0,255,0,150)}))

pyqtgraph.Qt.QtGui.QApplication.instance().exec_()
print "Done"
