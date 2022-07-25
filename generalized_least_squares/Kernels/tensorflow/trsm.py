import time
import tensorflow as tf


def tf_matmul_trsm(inp1, inp2, out):
    out = tf.matmul(inp2, tf.linalg.inv(inp1))
    # out = tf.matmul(inp2, tf.linalg.inv(tf.transpose(inp1)))
    # out = tf.matmul(tf.linalg.inv(inp1), inp2)
    # out = tf.matmul(tf.linalg.inv(tf.transpose(inp1)), inp2)
    return out


if __name__ == "__main__":
    # This function finds the solution to A*X = alpha*B
    # The are a total of 4 variations:
    # A could be in its initial form or transposed.
    # We could have either A*X or X*A
    # inp: for simplicity we assume for now that both input matrices are squared matrices.
    # inp1: Matrix A is a lower triangular matrix
    # inp2: Matrix B is a full rank matrix
    n = 100
    inp1 = tf.random.normal([n, n], dtype=tf.float64)
    inp1 = tf.linalg.band_part(inp1, -1, 0)
    inp2 = tf.random.normal([n, n], dtype=tf.float64)
    # out: size of the output must be equivalent to the size of te inputs.
    out = tf.zeros((n, n), dtype=tf.float64)

    start_time = time.perf_counter()
    tf_matmul_trsm(inp1, inp2, out)
    end_time = time.perf_counter()
    print(end_time - start_time)
