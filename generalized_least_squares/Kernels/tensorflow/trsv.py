import time
import tensorflow as tf


def tf_matmul_trsv(inp1, inp2, out):
    out = tf.matmul(tf.linalg.inv(inp1), inp2)
    return out


if __name__ == "__main__":
    # inp: for simplicity we assume for now that the input matrix is a squared matrix.
    # inverse of a lower triangular matrix in multiplied by a vector.
    # inp1: a lower triangular matrix
    # inp2: a vector
    n = 100
    inp1 = tf.random.normal([n, n], dtype=tf.float64)
    inp1 = tf.linalg.band_part(inp1, -1, 0)
    inp2 = tf.random.normal([n, 1], dtype=tf.float64)
    # out: the size of the output must be the same as the size of the input vector
    out = tf.random.normal([n], dtype=tf.float64)

    start_time = time.perf_counter()
    tf_matmul_trsv(inp1, inp2, out)
    end_time = time.perf_counter()
    print(end_time - start_time)
