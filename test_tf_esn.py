# test_tf_esn.py
import tensorflow as tf
import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot

import time

inSize = 1
outSize = 1
reservoirSize = 1000
unrolled_steps = 200
batch_size = 100
leaky = 0.3

mglass = numpy.loadtxt('MackeyGlass_t17.txt')
total_data_len = len(mglass)
training_data_len = int(0.8*total_data_len)
training_data = mglass[:training_data_len].reshape(1,-1)
testing_data = mglass[training_data_len:].reshape(1,-1)
# dataX = numpy.arange(-100*numpy.pi,100*numpy.pi,0.01).reshape(1,-1)
# total_length = dataX.shape[1]
# dataX = numpy.column_stack([samples,numpy.ones(shape=samples.shape)])
# dataY = numpy.sin(dataX)
# dataY = (dataX[:,0]*dataX[:,0]).reshape(-1,1)
# data = (dataX,dataY)

data = numpy.zeros(shape=(1000,inSize,unrolled_steps))
for i in range(len(data)):
	start = numpy.random.randint(training_data_len-unrolled_steps)
	data[i,:,:] = training_data[:,start:start+unrolled_steps]
train_layer1 = True
train_layer2 = False

#input, target output nodes.

# x = tf.placeholder(tf.float32, shape=[None,inSize])
# y_= tf.placeholder(tf.float32, shape=[None,outSize])
x = tf.placeholder(tf.float32,shape=[None,inSize,unrolled_steps])
# x = tf.placeholder(tf.float32,shape=[None,inSize])

Win = tf.Variable(tf.random_normal([inSize,reservoirSize],stddev=0.1),trainable=train_layer1)
Wout = tf.Variable(tf.random_normal([reservoirSize,outSize],stddev=0.1),trainable=train_layer2)
# Wr = tf.Variable(tf.random_normal([reservoirSize,reservoirSize],stddev=1.0),trainable=False)
Wr = numpy.random.rand(reservoirSize,reservoirSize)-0.5
print "computing spectral radius..."
rhoW = numpy.max(numpy.abs(numpy.linalg.eig(Wr)[0]))
print "done."
Wr = (Wr*1.25/rhoW).astype(numpy.float32)

unrolled_res = [leaky*tf.nn.tanh(tf.matmul(x[:,:,0],Win))]
loss = 0
# unrolled_res = [(1-leaky)*x + leaky*tf.nn.tanh(tf.matmul(x,Win))]
last_output = None
for i in range(1,unrolled_steps-1):
	tmp_ul_l = (1-leaky)*unrolled_res[-1]
	tmp_ul_ps = tf.matmul(unrolled_res[i-1],Wr)
	tmp_ul_ci = tf.matmul(x[:,:,i],Win)
	unrolled_res += [tmp_ul_l + leaky*tf.nn.tanh(tmp_ul_ps+tmp_ul_ci),]
	last_output = tf.matmul(unrolled_res[-1],Wout)
	if i > unrolled_steps/2:
		loss += tf.reduce_sum(tf.square(last_output-x[:,:,i+1]))

# y = tf.matmul(unrolled_res[-1],Wout)

# loss = tf.reduce_sum(tf.square(y-x[-1,:]))

#generative mode
test_outputs = [last_output]
test_states = [unrolled_res[-1]]
for i in range(unrolled_steps):
	new_state = (1-leaky)*test_states[-1] + leaky*tf.nn.tanh( tf.matmul(test_outputs[-1],Win) + tf.matmul(test_states[-1],Wr) )
	test_outputs += [tf.matmul(new_state,Wout),]
	test_states += [new_state,]
# testing_state = [(1-leaky)*x[-1,:] + leaky*tf.nn.tanh(tf.matmul(unrolled_res[-1],Wr)+tf.matmul())]
# for i in range(1,unrolled_steps):
# 
	# tmp_ul_l = (1-leaky)*test_ur[-1]
	# tmp_ul_ps = tf.matmul(test_ur[-1],Wr)
	# tmp_ul_ci = tf.matmul()

# Bin = tf.Variable(tf.random_normal([reservoirSize],stddev=0.1),trainable=train_layer1)
# h1 = tf.nn.tanh(tf.matmul(x,Win)+Bin)
# h1 = tf.nn.tanh(tf.matmul(x,Win))

# Wout = tf.Variable(tf.random_normal([reservoirSize,outSize],stddev=0.1),trainable = train_layer2)
# Bout = tf.Variable(tf.random_normal([outSize],stddev=0.1), trainable = train_layer2)
# y = tf.nn.tanh(tf.matmul(h1,Wout)+Bout)
# y = tf.nn.tanh(tf.matmul(h1,Wout))

# print tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
# exit()

# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	loss_list = []
	try:
		last_time = time.time()
		for i in range(10000):
			# bperm = numpy.random.permutation(len(data[0]))
			# fx = data[0][bperm][:100]
			# fy_ = data[1][bperm][:100]
			bperm = numpy.random.permutation(len(data))
			fx = data[bperm][:batch_size]
			# fy_ = data[bperm+1][:batch_size]
			if i%10==0:
				train_loss = loss.eval(feed_dict={x:fx})
				loss_list.append(train_loss)
				new_time = time.time()
				print "step {}, training loss {}".format(i,train_loss), "{} steps per second".format(10.0/float(new_time-last_time))
				last_time = new_time
			train_step.run(feed_dict={x:fx})
		# print sess.run([Win,Bin,Wout,Bout])
	except KeyboardInterrupt as kbi:
		print "Ctrl-c detected, stopping training (press again to quit)"
	if len(loss_list)>0:
		matplotlib.pyplot.plot(loss_list)
		try:
			matplotlib.pyplot.show()
		except:
			print "Could not display error curve"
		print "Saving error curve to 'training_error_curve.png'"
		matplotlib.pyplot.savefig('training_error_curve.png')
	# test_vis_data = numpy.column_stack(sess.run(test_outputs,feed_dict={x:data[numpy.random.permutation(len(data))][:5]}))
	# matplotlib.pyplot.plot(test_vis_data.T)
	# matplotlib.pyplot.show()
