import tensorflow as tf

"""
n = 10
A = tf.random.normal([n, n], dtype=tf.float64)
A = A + tf.transpose(A) + tf.eye(n, dtype=tf.float64) * n
B = tf.linalg.cholesky(A)
print(B)
"""

@tf.function
def run(X,M,y):

    stime0 = tf.timestamp()

    with tf.control_dependencies([stime0]):
        L7 = tf.linalg.cholesky(M)
    with tf.control_dependencies([L7]):
        stime1 = tf.timestamp()

    with tf.control_dependencies([stime1]):
        tmp57 = tf.matmul(tf.transpose(X), tf.linalg.inv(tf.transpose(L7)))
    with tf.control_dependencies([tmp57]):
        stime2 = tf.timestamp()

    with tf.control_dependencies([stime2]):
        tmp12 = tf.matmul(tf.linalg.inv(L7), X)
    with tf.control_dependencies([tmp12]):
        stime3 = tf.timestamp()

    with tf.control_dependencies([stime3]):
        tmp14 = tf.matmul(tf.transpose(tmp12), tmp12)
    with tf.control_dependencies([tmp14]):
        stime4 = tf.timestamp()

    with tf.control_dependencies([stime4]):
        L15 = tf.linalg.cholesky(tmp14)
    with tf.control_dependencies([L15]):
        stime5 = tf.timestamp()

    with tf.control_dependencies([stime5]):
        tmp68 = tf.matmul(tf.linalg.inv(L7), y)
    with tf.control_dependencies([tmp68]):
        stime6 = tf.timestamp()

    with tf.control_dependencies([stime6]):
        tmp21 = tf.matmul(tf.transpose(tmp12), tmp68)
    with tf.control_dependencies([tmp12]):
        stime7 = tf.timestamp()

    with tf.control_dependencies([stime7]):
        tmp23 = tf.matmul(tf.linalg.inv(L15), tmp21)
    with tf.control_dependencies([tmp23]):
        stime8 = tf.timestamp()

    with tf.control_dependencies([stime8]):
        tmp24 = tf.matmul(tf.linalg.inv(tf.transpose(L15)), tmp23)
    with tf.control_dependencies([tmp24]):
        stime9 = tf.timestamp()

    with tf.control_dependencies([stime9]):
        b = tmp24
    with tf.control_dependencies([b]):
        stime10 = tf.timestamp()


    timestamps = [stime0, stime1, stime2, stime3, stime4, stime5, stime6, stime7, stime8, stime9, stime10]

    return (b, timestamps)


def write_to_eventlog(csv_writer, exp_start_time, run_id, timestamps, dims, num_threads):
    id = "V1R" + str(run_id)
    # timestamps = [x*1e-9 for x in timestamps]
    timestamps = timestamps - exp_start_time

    event0 = [id, "(L7 L7^T) = M", timestamps[0].numpy(), timestamps[1].numpy(), dims, num_threads]
    csv_writer.writerow(event0)

    event1 = [id, "tmp57 = (X^T L7^-T)", timestamps[1].numpy(), timestamps[2].numpy(), dims, num_threads]
    csv_writer.writerow(event1)

    event2 = [id, "tmp12 = (L7^-1 X)", timestamps[2].numpy(), timestamps[3].numpy(), dims, num_threads]
    csv_writer.writerow(event2)

    event3 = [id, "tmp14 = (tmp12^T tmp12)", timestamps[3].numpy(), timestamps[4].numpy(), dims, num_threads]
    csv_writer.writerow(event3)

    event4 = [id, "(L15 L15^T) = tmp14", timestamps[4].numpy(), timestamps[5].numpy(), dims, num_threads]
    csv_writer.writerow(event4)

    event5 = [id, "tmp68 = (L7^-1 y)", timestamps[5].numpy(), timestamps[6].numpy(), dims, num_threads]
    csv_writer.writerow(event5)

    event6 = [id, "tmp21 = (tmp12^T tmp68)", timestamps[6].numpy(), timestamps[7].numpy(), dims, num_threads]
    csv_writer.writerow(event6)

    event7 = [id, "tmp23 = (L15^-1 tmp21)", timestamps[7].numpy(), timestamps[8].numpy(), dims, num_threads]
    csv_writer.writerow(event7)

    event8 = [id, "tmp24 = (L15^-T tmp23)", timestamps[8].numpy(), timestamps[9].numpy(), dims, num_threads]
    csv_writer.writerow(event8)

    event9 = [id, "b = tmp24", timestamps[9].numpy(), timestamps[10].numpy(), dims, num_threads]
    csv_writer.writerow(event9)



