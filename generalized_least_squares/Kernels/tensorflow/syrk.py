import time
import tensorflow as tf


def tf_matmul_syrk(inp, out):
    out = tf.matmul(inp, tf.transpose(inp))
    # out = tf.matmul(tf.transpose(inp), inp)
    return out


if __name__ == "__main__":
    # Rank-k update of the symmetric matrix C: C = alpha*A*transpose(A) + beta*C
    # inp: is a full-rank symmetric matrix which is multiplied with its transpose.
    # symmetric matrices are all squared matrices.
    n = 100
    inp = tf.random.normal([n, n], dtype=tf.float64)
    # out: the output is also a squared matrix which has the same size as the input matrix.
    out = tf.zeros((n, n), dtype=tf.float64)

    start_time = time.perf_counter()
    tf_matmul_syrk(inp, out)
    end_time = time.perf_counter()
    print(end_time - start_time)
