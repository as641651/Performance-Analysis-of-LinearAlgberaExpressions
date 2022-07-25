import time
import tensorflow as tf


def tf_linalg_cholesky(inp, out):
    out = tf.linalg.cholesky(inp)
    return out


if __name__ == "__main__":
    # inp: The input is a spd matrix.
    # The input is decomposed with Cholesky factorization.
    n = 100
    inp = tf.random.normal([n, n], dtype=tf.float64)
    inp = inp + tf.transpose(inp) + tf.eye(n, dtype=tf.float64) * n
    # out: The out put is a lower triangular matrix
    out = tf.zeros((n, n), dtype=tf.float64)

    start_time = time.perf_counter()
    tf_linalg_cholesky(inp, out)
    end_time = time.perf_counter()
    print(end_time - start_time)

