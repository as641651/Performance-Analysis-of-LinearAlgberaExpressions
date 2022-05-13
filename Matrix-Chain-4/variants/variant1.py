import tensorflow as tf
import utils
#import csv


@tf.function
def run(A,B,C,D):
    
    stime0 = tf.timestamp()

    with tf.control_dependencies([stime0]):
        T_AB = tf.matmul(A,B)
    with tf.control_dependencies([T_AB]):
        stime1 = tf.timestamp()

    with tf.control_dependencies([stime1]):
        T_ABC = tf.matmul(T_AB, C)
    with tf.control_dependencies([T_ABC]):
        stime2 = tf.timestamp() 

    with tf.control_dependencies([stime2]):
        Y = tf.matmul(T_ABC, D)
    with tf.control_dependencies([Y]):
        stime3 = tf.timestamp() 

    timestamps = [stime0,stime1,stime2, stime3]

    return (Y,timestamps)
    

def write_to_eventlog(csv_writer,run_id,timestamps,dims,num_threads):
    id = "V1R"+str(run_id)
    #timestamps = [x*1e-9 for x in timestamps]
    #timestamps =  exp_start_time

    event0 = [id, "matmul(A,B)", utils.convert_timestamp_todtime(timestamps[0].numpy()), utils.convert_timestamp_todtime(timestamps[1].numpy()), dims, num_threads]
    csv_writer.writerow(event0)

    event1 = [id, "matmul(T_AB,C)", utils.convert_timestamp_todtime(timestamps[1].numpy()), utils.convert_timestamp_todtime(timestamps[2].numpy()), dims, num_threads]
    csv_writer.writerow(event1)

    event2 = [id, "matmul(T_ABC,D)", utils.convert_timestamp_todtime(timestamps[2].numpy()), utils.convert_timestamp_todtime(timestamps[3].numpy()), dims, num_threads]
    csv_writer.writerow(event2)






