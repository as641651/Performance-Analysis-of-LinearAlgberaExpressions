import tensorflow as tf


# import csv


@tf.function
def run(X,M,y):
    ml0 = X
    ml1 = M
    ml2 = y
    stime0 = tf.timestamp()

    with tf.control_dependencies([stime0]):
        ml1 = tf.linalg.cholesky(ml1)
    with tf.control_dependencies([ml1]):
        stime1 = tf.timestamp()

    with tf.control_dependencies([stime1]):
        ml2 = tf.matmul(tf.linalg.inv(ml1), ml2)
    with tf.control_dependencies([ml2]):
        stime2 = tf.timestamp()

    with tf.control_dependencies([stime2]):
        ml0 = tf.matmul(tf.linalg.inv(ml1), ml0)
    with tf.control_dependencies([ml0]):
        stime3 = tf.timestamp()

    with tf.control_dependencies([stime3]):
        ml3 = tf.matmul(tf.transpose(ml0), ml0)
    with tf.control_dependencies([ml3]):
        stime4 = tf.timestamp()

    with tf.control_dependencies([stime4]):
        ml4 = tf.matmul(tf.transpose(ml0), ml2)
    with tf.control_dependencies([ml4]):
        stime5 = tf.timestamp()

    with tf.control_dependencies([stime5]):
        ml3 = tf.linalg.cholesky(ml3)
    with tf.control_dependencies([ml3]):
        stime6 = tf.timestamp()

    with tf.control_dependencies([stime6]):
        ml4 = tf.matmul(tf.linalg.inv(ml3), ml4)
    with tf.control_dependencies([ml4]):
        stime7 = tf.timestamp()

    with tf.control_dependencies([stime7]):
        ml4 = tf.matmul(tf.linalg.inv(ml3), ml4)
    with tf.control_dependencies([ml4]):
        stime8 = tf.timestamp()



    timestamps = [stime0,stime1,stime2,stime3,stime4,stime5,stime6,stime7,stime8]

    return (ml4, timestamps)


def write_to_eventlog(csv_writer, exp_start_time, run_id, timestamps, dims, num_threads):
    id = "V1R" + str(run_id)
    # timestamps = [x*1e-9 for x in timestamps]
    timestamps = timestamps - exp_start_time

    event0 = [id, "LAPACK.potrf", timestamps[0].numpy(), timestamps[1].numpy(), dims, num_threads]
    csv_writer.writerow(event0)

    event1 = [id, "trsv", timestamps[1].numpy(), timestamps[2].numpy(), dims, num_threads]
    csv_writer.writerow(event1)

    event2 = [id, "trsm", timestamps[2].numpy(), timestamps[3].numpy(), dims, num_threads]
    csv_writer.writerow(event2)

    event3 = [id, "syrk", timestamps[3].numpy(), timestamps[4].numpy(), dims, num_threads]
    csv_writer.writerow(event3)

    event4 = [id, "gemv", timestamps[4].numpy(), timestamps[5].numpy(), dims, num_threads]
    csv_writer.writerow(event4)

    event5 = [id, "LAPACK.potrf", timestamps[5].numpy(), timestamps[6].numpy(), dims, num_threads]
    csv_writer.writerow(event5)

    event6 = [id, "trsv", timestamps[6].numpy(), timestamps[7].numpy(), dims, num_threads]
    csv_writer.writerow(event6)

    event7 = [id, "trsv", timestamps[7].numpy(), timestamps[8].numpy(), dims, num_threads]
    csv_writer.writerow(event7)










