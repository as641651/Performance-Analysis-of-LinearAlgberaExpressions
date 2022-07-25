import time
import tensorflow as tf


def tf_matmul_gemm(inp1, inp2, out):
    out = tf.matmul(inp1, inp2)
    # out = tf.matmul(tf.transpose(inp1), inp2)
    # out = tf.matmul(inp1, tf.transpose(inp2))
    # out = tf.matmul(tf.transpose(inp1), tf.transpose(inp2))
    return out


if __name__ == "__main__":
    # This function multiplies two full rank matrices. Both matrices could be in their initial form o transposed.
    # for now, we assume that both inputs are square matrices --> I do not know if this assumption is necessary or not!
    # how should we create a full rank matrix for this experiment?
    n = 100
    inp1 = tf.random.normal([n, n], dtype=tf.float64)
    inp2 = tf.random.normal([n, n], dtype=tf.float64)
    # output: The size of the output is the same as the size of inputs.
    out = tf.zeros((n, n), dtype=tf.float64)

    start_time = time.perf_counter()
    tf_matmul_gemm(inp1, inp2, out)
    end_time = time.perf_counter()
    print(end_time - start_time)
