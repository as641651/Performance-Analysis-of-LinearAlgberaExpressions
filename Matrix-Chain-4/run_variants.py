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
header = ['ID', 'Operation', 'Start time', 'End time', 'Dims', 'Num Threads']
csv_writer.writerow(header)

# TODO move dimensions inside for loop and choose sampling strategy
m = 3000
n = 3000
k = 3000
l = 3000
q = 3000

A = tf.random.normal([m, n], dtype=DTYPE)
B = tf.random.normal([n, k], dtype=DTYPE)
C = tf.random.normal([k, l], dtype=DTYPE)
D = tf.random.normal([l, q], dtype=DTYPE)


import variants.variant1 as v1
import variants.variant2 as v2
import variants.variant3 as v3
import variants.variant4 as v4
import variants.variant5 as v5

exp_start_time = tf.timestamp()

for i in range(REPS):
    Y,timestamps = v1.run(A,B,C,D)
    v1.write_to_eventlog(csv_writer,exp_start_time,i,timestamps,[m,n,k,l,q],NUM_THREADS)

print("Variant 1 done")

for i in range(REPS):
    Y,timestamps = v2.run(A,B,C,D)
    v2.write_to_eventlog(csv_writer,exp_start_time,i,timestamps,[m,n,k,l,q],NUM_THREADS)

print("Variant 2 done")

for i in range(REPS):
    Y,timestamps = v3.run(A,B,C,D)
    v3.write_to_eventlog(csv_writer,exp_start_time,i,timestamps,[m,n,k,l,q],NUM_THREADS)

print("Variant 3 done")

for i in range(REPS):
    Y,timestamps = v4.run(A,B,C,D)
    v4.write_to_eventlog(csv_writer,exp_start_time,i,timestamps,[m,n,k,l,q],NUM_THREADS)

print("Variant 4 done")

for i in range(REPS):
    Y,timestamps = v5.run(A,B,C,D)
    v5.write_to_eventlog(csv_writer,exp_start_time,i,timestamps,[m,n,k,l,q],NUM_THREADS)

print("Variant 5 done")

f.close()



