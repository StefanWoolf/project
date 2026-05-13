import os
import random
import numpy as np

SEED = 29042005


def set_seeds(seed: int = SEED) -> None:

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def set_seeds_tf(seed: int = SEED) -> None:
    set_seeds(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:

            pass
    except ImportError:
        pass 