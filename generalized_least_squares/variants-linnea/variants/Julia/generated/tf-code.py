import tensorflow as tf


# import csv


@tf.function
def run(X,M,y):
    ml0 = X
    ml1 = M
    ml2 = y
    stime0 = tf.timestamp()

{code}

    timestamps = {timestamps}

    return ({result}, timestamps)


def write_to_eventlog(csv_writer, exp_start_time, run_id, timestamps, dims, num_threads):
    id = "V1R" + str(run_id)
    # timestamps = [x*1e-9 for x in timestamps]
    timestamps = timestamps - exp_start_time

{code1}








