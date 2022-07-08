import time
import tensorflow as tf

if __name__ == "__main__":
    # this function only transposes the input matrix.
    n = 100
    inp = tf.random.normal([n, n], dtype=tf.float64)

    start_time = time.perf_counter()
    inp = tf.transpose(inp)
    end_time = time.perf_counter()
    print(end_time - start_time)
