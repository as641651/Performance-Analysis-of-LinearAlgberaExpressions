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
n = 500
m = 2500

# b = tf.random.normal([n, 1], dtype=DTYPE)
X = tf.random.normal([m, n], dtype=DTYPE)
M = tf.random.normal([m, m], dtype=DTYPE)
M = M + tf.transpose(M) + tf.eye(m, dtype=DTYPE) * m
y = tf.random.normal([m, 1], dtype=DTYPE)


import variants.variant1 as v1
import variants.variant2 as v2
import variants.variant3 as v3
import variants.variant4 as v4
import variants.variant5 as v5

exp_start_time = tf.timestamp()

for i in range(REPS):
    b,timestamps = v1.run(X,M,y)
    v1.write_to_eventlog(csv_writer,exp_start_time,i,timestamps,[m,n],NUM_THREADS)

print("Variant 1 done")


for i in range(REPS):
    b, timestamps = v2.run(X, M, y)
    v2.write_to_eventlog(csv_writer, exp_start_time, i, timestamps, [m, n], NUM_THREADS)

print("Variant 2 done")

for i in range(REPS):
    b, timestamps = v3.run(X, M, y)
    v3.write_to_eventlog(csv_writer, exp_start_time, i, timestamps, [m, n], NUM_THREADS)

print("Variant 3 done")

for i in range(REPS):
    b, timestamps = v4.run(X, M, y)
    v4.write_to_eventlog(csv_writer, exp_start_time, i, timestamps, [m, n], NUM_THREADS)

print("Variant 4 done")

for i in range(REPS):
    b, timestamps = v5.run(X, M, y)
    v5.write_to_eventlog(csv_writer, exp_start_time, i, timestamps, [m, n], NUM_THREADS)

print("Variant 5 done")

f.close()

