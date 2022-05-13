import tensorflow as tf
import csv
import utils
import post_process
import time

class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'


#Check if MKL is enabled
import tensorflow.python.framework as tff
print(bcolors.WARNING + "MKL Enabled : ", tff.test_util.IsMklEnabled(), bcolors.ENDC)


#Set threads
NUM_THREADS = 4
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
tf.config.run_functions_eagerly(False)
DTYPE = tf.float32
REPS = 70


event_log = 'logs/event_log-3.csv'
deviations_log = 'logs/deviations-3.csv'
# log file
f = open(event_log, 'w',encoding='UTF8')
csv_writer = csv.writer(f)
header = ['case:concept:name', 'concept:name', 'time:start', 'time:end', 'case:dims', 'case:threads']
csv_writer.writerow(header)

# TODO move dimensions inside for loop and choose sampling strategy
m = 300
n = 4000
k = 30
l = 320
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

variants = [v1,
            v2,
            v3,
            v4,
            v5
            ]

measurements_instance_set = utils.get_shuffled_measurements_instance_set(variants, REPS)

c = 0
for id, instance in measurements_instance_set:

    c = c+1
    #if c>100 and c <150:
    #    time.sleep(0.1)
    print(id, instance)
    Y,timestamps = instance.run(A,B,C,D)
    print(timestamps[-1] - timestamps[0])
    instance.write_to_eventlog(csv_writer,id,timestamps,[m,n,k,l,q],NUM_THREADS)


f.close()

variants_str = ['V1', 'V2', 'V3', 'V4', 'V5']
post_process.add_duration_deviations(event_log,deviations_log,variants_str,0.6)


