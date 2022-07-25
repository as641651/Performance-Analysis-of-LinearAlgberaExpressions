import time
import tensorflow as tf


def tf_matmul_gemv(inp1,inp2 ,out):
    out = tf.matmul(tf.transpose(inp1), inp2)
    # out = tf.matmul(inp1, inp2)
    return out


if __name__ == "__main__":
    # This function multiplies a full rank matrix and a vector.
    # The matrix could be in its initial form or transposed.
    # For now, we assume that the input matrix is a squared matrix
    # --> I do not know if this assumption is necessary or not!
    # how should we create a full rank matrix for this experiment?
    n = 100
    inp1 = tf.random.normal([n, n], dtype=tf.float64)
    inp2 = tf.random.normal([n, 1], dtype=tf.float64)
    # output: The size of the output is the same as the type of the input.
    out = tf.zeros((n, 1), dtype=tf.float64)

    start_time = time.perf_counter()
    tf_matmul_gemv(inp1, inp2, out)
    end_time = time.perf_counter()
    print(end_time - start_time)
