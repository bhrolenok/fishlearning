import numpy, btfutil, cPickle, scipy.stats
import pyqtgraph, pyqtgraph.opengl
# import matplotlib.pyplot
import sys

pyqtgraph.setConfigOption('background','w')
pyqtgraph.setConfigOption('foreground','k')
# mw = pyqtgraph.Qt.QtGui.QMainWindow()
# view = pyqtgraph.GraphicsLayoutWidget()
# mw.setCentralWidget(view)
# mw.show()

# w1 = view.addPlot()

# w = pyqtgraph.opengl.GLViewWidget()
# w.resize(1800,900)
# w.show()

EPS = 0.0000001

# seg_file_path = '../../fish_data/pickles/simsigma_0.15_pvel.p'
# sim_file_path = '../../Desktop/FishLRLogger-knnsamplenorm-july-07-2016-13.26/'
# sim_file_path = '../../Desktop/FishLRLogger-knnaverage-july-07-2016-14.04/'
# sim_file_path = '../../Desktop/FishLRLogger-linreg-july-07-2016-14.40/'
cv_hist_path = sys.argv[1]
learn_hist_path = sys.argv[2]
print "Loading histograms"
# res = cPickle.load(open(seg_file_path))
learn_hist = numpy.genfromtxt(learn_hist_path)
cv_hist = numpy.genfromtxt(cv_hist_path)
xline = numpy.arange(-0.5,0.5+(1.0/float(len(cv_hist))),1.0/float(len(cv_hist)))
kldiv = scipy.stats.entropy(cv_hist,learn_hist+EPS)
print "KL divergence for x-velocity:", kldiv

#combo xVel
w_combo = pyqtgraph.GraphicsWindow(title="Combined xVel")
plt_c = w_combo.addPlot()
plt_c.plot(xline,cv_hist, stepMode=True, fillLevel=0, brush=(0,0,255,150))
# plt_c.addItem(pyqtgraph.InfiniteLine(xvel_mean,pen={'style':1,'color':(50,50,50,200)}))
# plt_c.addItem(pyqtgraph.InfiniteLine(xvel_mean+xvel_std,pen={'style':3,'color':(50,50,50,200)}))
# plt_c.addItem(pyqtgraph.InfiniteLine(xvel_mean-xvel_std,pen={'style':3,'color':(50,50,50,200)}))
plt_c.plot(xline,learn_hist,stepMode=True,fillLevel=0,brush=(0,150,0,200))
# plt_c.addItem(pyqtgraph.InfiniteLine(sim_xvel_mean,pen={'style':1,'color':(0,150,0,200)}))
# plt_c.addItem(pyqtgraph.InfiniteLine(sim_xvel_mean+sim_xvel_std,pen={'style':3,'color':(0,150,0,200)}))
# plt_c.addItem(pyqtgraph.InfiniteLine(sim_xvel_mean-sim_xvel_std,pen={'style':3,'color':(0,150,0,200)}))
# plt_c.addItem(pyqtgraph.TextItem("KL divergence: {}".format(kldiv),color=(0,0,0)))

pyqtgraph.Qt.QtGui.QApplication.instance().exec_()
print "Done"
