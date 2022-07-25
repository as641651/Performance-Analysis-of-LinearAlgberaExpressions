import time
import tensorflow as tf


def tf_linalg_lu(inp, out1, out2):
    out1, out2 = tf.linalg.lu(inp)
    return out1, out2


if __name__ == "__main__":
    # This function the input matrix with its pivoted LU factorization.
    # I did not find the exact equivalent of this kernel in tensorflow.
    # For this reason I used the only existing alternative: The simple lu factorization.
    # inp: The input is a matrix: For simplicity, we assume that this matrix is an squared matrix.
    n = 100
    inp = tf.random.normal([n, n], dtype=tf.float64)
    # out1 : The first output "lu" has the same size as the input matrix: I guess this is true for squared matrices.
    # out2: the second output is the permutation matrix that is used in the process.
    #   I am not sure if the second output has the same size as the input matrix.
    out1 = tf.zeros((n, n), dtype=tf.float64)
    out2 = tf.zeros((n, n), dtype=tf.float64)

    start_time = time.perf_counter()
    tf_linalg_lu(inp, out1, out2)
    end_time = time.perf_counter()
    print(end_time - start_time)
