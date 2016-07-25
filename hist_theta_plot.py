# hist_theta_plot.py
import numpy, matplotlib.pyplot, btfutil, cPickle, scipy.stats
import pyqtgraph, pyqtgraph.opengl
import sys

EPS = 0.0000001

real_file_path = '../../fish_data/java_btf_seqs/'
sim_file_path = '../../Desktop/FishLRLogger-momentum-july-12/'
if len(sys.argv) > 1 and len(sys.argv) < 4:
	sim_file_path = sys.argv[1]
	if len(sys.argv) == 3:
		real_file_path = sys.argv[2]
print "Loading segments"
res = btfutil.load_sequence_dir(real_file_path)
sim_res = btfutil.BTF()
sim_res.import_from_dir(sim_file_path)
print "Stacking data"
fnames = ['rbfsepvec', 'rbforivec','rbfcohvec', 'rbfwallvec','pvel']
feats_and_outs = map(lambda r: btfutil.btf2data(r,fnames,False),res)
all_outs = numpy.row_stack([thing[1] for thing in feats_and_outs])
sim_outs = btfutil.btf2data(sim_res,fnames,False)[1]
all_outs = all_outs[:len(sim_outs)]

print "Creating histograms"
# percentage is a bit hard to tell, so use raw counts
tvel_y, tvel_x = numpy.histogram(all_outs[:,2],bins=150)
sim_tvel_y, sim_tvel_x = numpy.histogram(sim_outs[:,2],bins=150)
# chisquared requires sum=1.0
# tvel_y, tvel_x = numpy.histogram(all_outs[:,2],bins=150, density=True)
# sim_tvel_y, sim_tvel_x = numpy.histogram(sim_outs[:,2],bins=150, density=True)
cstest_res = scipy.stats.chisquare(tvel_y/tvel_y.sum(),(sim_tvel_y/sim_tvel_y.sum())+EPS)
print "Chi-square test results for x-velocity:", cstest_res

print "Plotting"
app = pyqtgraph.Qt.QtGui.QApplication([])
#tVel
w1 = pyqtgraph.GraphicsWindow(title="tvel")
plt1 = w1.addPlot()
plt1.plot(tvel_x, tvel_y,stepMode=True, fillLevel=0,brush=(0,0,255,150))
tvel_mean = all_outs[:,2].mean()
tvel_std = all_outs[:,2].std()
print "tvel mean +- std:", tvel_mean,"+-",tvel_std
plt1.addItem(pyqtgraph.InfiniteLine(tvel_mean,pen={'style':1,'color':(255,255,255,150)}))
plt1.addItem(pyqtgraph.InfiniteLine(tvel_mean+tvel_std,pen={'style':3,'color':(255,255,255,150)}))
plt1.addItem(pyqtgraph.InfiniteLine(tvel_mean-tvel_std,pen={'style':3,'color':(255,255,255,150)}))
plt1.addItem(pyqtgraph.TextItem("{} +- {}".format(tvel_mean,tvel_std)))

#sim tVel
w_sim = pyqtgraph.GraphicsWindow(title="Sim tVel")
plt_sim = w_sim.addPlot()
plt_sim.plot(sim_tvel_x, sim_tvel_y,stepMode=True,fillLevel=0,brush=(0,255,0,150))
sim_tvel_mean = sim_outs[:,2].mean()
sim_tvel_std = sim_outs[:,2].std()
print "sim tvel mean +- std:", sim_tvel_mean,"+-",sim_tvel_std
plt_sim.addItem(pyqtgraph.InfiniteLine(sim_tvel_mean,pen={'style':1,'color':(0,255,0,150)}))
plt_sim.addItem(pyqtgraph.InfiniteLine(sim_tvel_mean+sim_tvel_std,pen={'style':3,'color':(0,255,0,150)}))
plt_sim.addItem(pyqtgraph.InfiniteLine(sim_tvel_mean-sim_tvel_std,pen={'style':3,'color':(0,255,0,150)}))
plt_sim.addItem(pyqtgraph.TextItem("{} +- {}".format(sim_tvel_mean,sim_tvel_std)))

#combo tVel
w_combo = pyqtgraph.GraphicsWindow(title="Combined tVel")
plt_c = w_combo.addPlot()
plt_c.plot(tvel_x, tvel_y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
plt_c.addItem(pyqtgraph.InfiniteLine(tvel_mean,pen={'style':1,'color':(255,255,255,150)}))
plt_c.addItem(pyqtgraph.InfiniteLine(tvel_mean+tvel_std,pen={'style':3,'color':(255,255,255,150)}))
plt_c.addItem(pyqtgraph.InfiniteLine(tvel_mean-tvel_std,pen={'style':3,'color':(255,255,255,150)}))
plt_c.plot(sim_tvel_x, sim_tvel_y,stepMode=True,fillLevel=0,brush=(0,255,0,150))
plt_c.addItem(pyqtgraph.InfiniteLine(sim_tvel_mean,pen={'style':1,'color':(0,255,0,150)}))
plt_c.addItem(pyqtgraph.InfiniteLine(sim_tvel_mean+sim_tvel_std,pen={'style':3,'color':(0,255,0,150)}))
plt_c.addItem(pyqtgraph.InfiniteLine(sim_tvel_mean-sim_tvel_std,pen={'style':3,'color':(0,255,0,150)}))
plt_c.addItem(pyqtgraph.InfiniteLine(numpy.pi*2.0/(1.0/30.0),pen={'style':2,'color':(255,255,255,100)}))
plt_c.addItem(pyqtgraph.InfiniteLine(-numpy.pi*2.0/(1.0/30.0),pen={'style':2,'color':(255,255,255,100)}))
plt_c.addItem(pyqtgraph.InfiniteLine(2*numpy.pi*2.0/(1.0/30.0),pen={'style':2,'color':(255,255,255,100)}))
plt_c.addItem(pyqtgraph.InfiniteLine(-2*numpy.pi*2.0/(1.0/30.0),pen={'style':2,'color':(255,255,255,100)}))
plt_c.addItem(pyqtgraph.InfiniteLine(3*numpy.pi*2.0/(1.0/30.0),pen={'style':2,'color':(255,255,255,100)}))
plt_c.addItem(pyqtgraph.InfiniteLine(-3*numpy.pi*2.0/(1.0/30.0),pen={'style':2,'color':(255,255,255,100)}))
# right_axes = pyqtgraph.AxisItem(orientation='right',linkView=plt_c.getViewBox(),parent=w_combo)
# right_axes.setScale(1.0/sum(tvel_y))
# plt_c.addItem(right_axes)
plt_c.addItem(pyqtgraph.TextItem("pval: {}, stat: {}".format(cstest_res[1],cstest_res[0])))

pyqtgraph.Qt.QtGui.QApplication.instance().exec_()
print "Done"
