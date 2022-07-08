import time
import tensorflow as tf


def VecInv(inp1, inp2, out):
    out = tf.matmul(tf.linalg.inv(inp1), inp2)
    return out


if __name__ == "__main__":
    # This function multiplies the inverse of a vector with a matrix and overwrites the matrix.
    # inp : The two inputs are a diagonal matrix and a full matrix:
    n = 100
    inp1 = tf.random.normal([n, n], dtype=tf.float64)
    inp1 = tf.linalg.diag(tf.linalg.diag_part(inp1))
    inp2 = tf.random.normal([n, n], dtype=tf.float64)
    # The output has the same size as the inputs.
    out = tf.zeros((n, n), dtype=tf.float64)

    start_time = time.perf_counter()
    VecInv(inp1, inp2, out)
    end_time = time.perf_counter()
    print(end_time - start_time)

