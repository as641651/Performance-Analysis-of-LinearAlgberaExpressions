import time
import tensorflow as tf


def tf_linalg_cholesky(inp, out):
    out = tf.linalg.cholesky(inp)
    return out


if __name__ == "__main__":
    # input must be an spd matrix
    n = 100
    inp = tf.random.normal([n, n], dtype=tf.float64)
    inp = inp + tf.transpose(inp) + tf.eye(n, dtype=tf.float64) * n
    # output has the same size as input
    out = tf.zeros((n, n), dtype=tf.float64)

    start_time = time.perf_counter()
    tf_linalg_cholesky(inp, out)
    end_time = time.perf_counter()
    print(end_time - start_time)
