import time
import tensorflow as tf


def tf_linalg_lu(inp, out1, out2):
    out1, out2 = tf.linalg.lu(inp)
    return out1, out2


if __name__ == "__main__":
    # This function was not included in the linea variants.
    # Could be modified later
    # input and outputs must be defined to be passed into the function.

    start_time = time.perf_counter()
    tf_linalg_lu(inp, out1, out2)
    end_time = time.perf_counter()
    print(end_time - start_time)
