import tensorflow as tf
#import csv


@tf.function
def run(A,B,C,D):
    
    stime0 = tf.timestamp()

    with tf.control_dependencies([stime0]):
        T_BC = tf.matmul(B,C)
    with tf.control_dependencies([T_BC]):
        stime1 = tf.timestamp() 
    
    with tf.control_dependencies([stime1]):
        T_BCD = tf.matmul(T_BC,D)
    with tf.control_dependencies([T_BCD]):
        stime2 = tf.timestamp() 

    with tf.control_dependencies([stime2]):
        Y = tf.matmul(A, T_BCD)
    with tf.control_dependencies([Y]):
        stime3 = tf.timestamp() 

    timestamps = [stime0,stime1,stime2, stime3]

    return (Y,timestamps)
    

def write_to_eventlog(csv_writer,exp_start_time,run_id,timestamps,dims,num_threads):
    id = "V3R"+str(run_id)
    #timestamps = [x*1e-9 for x in timestamps]
    timestamps = timestamps - exp_start_time

    event0 = [id, "matmul(B,C)", timestamps[0].numpy(), timestamps[1].numpy(), dims, num_threads]
    csv_writer.writerow(event0)

    event1 = [id, "matmul(T_BC,D)", timestamps[1].numpy(), timestamps[2].numpy(), dims, num_threads]
    csv_writer.writerow(event1)

    event2 = [id, "matmul(A,T_BCD)", timestamps[2].numpy(), timestamps[3].numpy(), dims, num_threads]
    csv_writer.writerow(event2)






