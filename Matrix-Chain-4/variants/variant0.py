import tensorflow as tf
import csv


@tf.function
def run(A,B,C,D):
    
    stime0 = tf.timestamp()

    T_AB = tf.matmul(A,B)
    stime1 = tf.timestamp() 
    
    T_ABC = tf.matmul(T_AB, C)
    stime2 = tf.timestamp() 

    Y = tf.matmul(T_ABC, D)
    stime3 = tf.timestamp() 

    timestamps = [stime0,stime1,stime2, stime3]

    return (Y,timestamps)
    

def write_to_eventlog(csv_writer,run_id,timestamps,dims,num_threads):
    id = "V0R"+str(run_id)
    timestamps = timestamps - timestamps[0]

    event0 = [id, "matmul(A,B)", timestamps[1].numpy(), dims, num_threads]
    csv_writer.writerow(event0)

    event1 = [id, "matmul(T_AB,C)", timestamps[2].numpy(), dims, num_threads]
    csv_writer.writerow(event1)

    event2 = [id, "matmul(T_ABC,D)", timestamps[3].numpy(), dims, num_threads]
    csv_writer.writerow(event2)






