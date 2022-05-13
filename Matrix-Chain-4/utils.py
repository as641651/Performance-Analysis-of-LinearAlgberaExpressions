from datetime import datetime
import random
import pandas as pd

def convert_timestamp_todtime(timestamp):
    #return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
    return timestamp


def get_shuffled_measurements_instance_set(variants, reps):
    """TODO: doc"""
    measurements_instance_set = []
    for variant in variants:
        measurements_instance_set = measurements_instance_set + [(i,variant) for i in range(reps)]

    random.shuffle(measurements_instance_set)
    return measurements_instance_set








