import tensorflow as tf
import csv

class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'


#Check if MKL is enabled
import tensorflow.python.framework as tff
print(bcolors.WARNING + "MKL Enabled : ", tff.test_util.IsMklEnabled(), bcolors.ENDC)


#Set threads
NUM_THREADS = 1
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
tf.config.run_functions_eagerly(False)
DTYPE = tf.float32
REPS = 5

# log file
f = open('logs/event_log.csv', 'w',encoding='UTF8')
csv_writer = csv.writer(f)
header = ['ID', 'Operation', 'Start time', 'Dims', 'Num Threads']
csv_writer.writerow(header)

# TODO move dimensions inside for loop and choose sampling strategy
m = 500
n = 500
k = 500
l = 500
q = 500

A = tf.random.normal([m, n], dtype=DTYPE)
B = tf.random.normal([n, k], dtype=DTYPE)
C = tf.random.normal([k, l], dtype=DTYPE)
D = tf.random.normal([l, q], dtype=DTYPE)


import variants.variant0 as v0

for i in range(REPS):
    Y,timestamps = v0.run(A,B,C,D)
    v0.write_to_eventlog(csv_writer,i,timestamps,[m,n,k,l,q],NUM_THREADS)

f.close()



