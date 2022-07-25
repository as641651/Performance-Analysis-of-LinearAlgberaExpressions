import time
import tensorflow as tf


def tf_linalg_eigh(inp, out1, out2):
    out1, out2 = tf.linalg.eigh(inp)
    return out1, out2


if __name__ == "__main__":
    # inp: The input matrix is a symmetric squared matrix
    # This function applies eigen-decomposition of the input matrix
    n = 100
    inp = tf.random.normal([n, n], dtype=tf.float64)
    inp = tf.matmul(inp, tf.transpose(inp))
    # out:
    # out1: The first output is a vector containing all eigenvalues of the matrix.
    # This vector has n entries.
    # out2: The second output is a matrix containing the eigenvectors of the input matrix.
    # It hats the same size as the input matrix.
    out1 = tf.zeros(n, dtype=tf.float64)
    out2 = tf.zeros((n, n), dtype=tf.float64)

    start_time = time.perf_counter()
    tf_linalg_eigh(inp, out1, out2)
    end_time = time.perf_counter()
    print(end_time - start_time)
