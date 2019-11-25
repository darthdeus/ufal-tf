import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    import tensorflow as tf
    intra3 = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1,
            use_per_session_threads=True,
            log_device_placement=False,
            intra_op_parallelism_threads=111,
            allow_soft_placement=True)

    intra5 = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1,
            use_per_session_threads=True,
            log_device_placement=False,
            intra_op_parallelism_threads=255,
            allow_soft_placement=True)

    import ipdb

    serialized3 = intra3.SerializeToString()
    print("3: {}".format(serialized3))
    print(", ".join("{:02x}".format(ord(c)) for c in serialized3.decode("ascii")))
    parsed3 = tf.compat.v1.ConfigProto().FromString(serialized3)

    serialized5 = intra5.SerializeToString()
    print("5: {}".format(serialized5))
    parsed5 = tf.compat.v1.ConfigProto().FromString(serialized5)

    ipdb.set_trace()

    print(parsed5)
