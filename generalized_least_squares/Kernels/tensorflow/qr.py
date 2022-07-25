import time
import tensorflow as tf


def tf_linalg_qr(inp, out1, out2):
    out1, out2= tf.linalg.qr(inp)
    return inp


if __name__ == "__main__":
    # This function computes the qr factorization of the input matrix.
    # inp: The input matrix is not required at all to be a squared matrix.
    # However, for simplicity, here we assume that the input matrix is an squared matrix of size n*n.
    n = 100
    inp = tf.random.normal([n, n], dtype=tf.float64)
    # out: Here, since we assumed that the input matrix is a squared matrix, the two output matrices will also be
    #   of the same size.
    out1 = tf.zeros((n, n), dtype=tf.float64)
    out2 = tf.zeros((n, n), dtype=tf.float64)

    start_time = time.perf_counter()
    tf_linalg_qr(inp, out1, out2)
    end_time = time.perf_counter()
    print(end_time - start_time)
