"""Общие утилиты проекта."""
import os
import random
import numpy as np

SEED = 29042005


def set_seeds(seed: int = SEED) -> None:
    """Зафиксировать случайные сиды во всех используемых библиотеках."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # TensorFlow и torch добавим в более поздних шагах,
    # чтобы сейчас не тянуть тяжёлые импорты


def set_seeds_tf(seed: int = SEED) -> None:
    """Зафиксировать случайные сиды для TensorFlow + базовые библиотеки.

    Должна вызываться ДО создания любых слоёв Keras.
    Требует TensorFlow >= 2.8.
    """
    set_seeds(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            # enable_op_determinism доступна с TF 2.10+; игнорируем на старых версиях
            pass
    except ImportError:
        pass  # TF не установлен — вызов set_seeds_tf без TF не ошибка