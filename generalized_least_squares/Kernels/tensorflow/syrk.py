import time
import tensorflow as tf


def tf_matmul_syrk(inp1, inp2, out):
    out = tf.matmul(tf.transpose(inp1), inp2)
    return out


if __name__ == "__main__":
    # inp: is a full-rank matrix which is multiplied with its transpose.
    # for now, we assume that the input matrix is a squared matrix.
    n = 100
    inp1 = tf.random.normal([n, n], dtype=tf.float64)
    inp2 = inp1
    # out: the output is also a squared matrix which has the same size as the input matrix.
    out = tf.zeros((n, n), dtype=tf.float64)

    start_time = time.perf_counter()
    tf_matmul_syrk(inp1, inp2, out)
    end_time = time.perf_counter()
    print(end_time - start_time)
