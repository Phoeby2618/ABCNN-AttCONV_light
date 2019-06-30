import tensorflow as tf

def mask_softmax(logits, mask=None, scope=None):
    with tf.name_scope(scope or "MaskSoftmax"):
        if mask is not None:
            logits = exp_mask(logits, mask)
        flat_out = tf.nn.softmax(logits)
        return flat_out

def exp_mask(val, mask, name=None):
        """Give very negative number to unmasked elements in val.
        For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
        Typically, this effectively masks in exponential space (e.g. softmax)
        Args:
            val: values to be masked
            mask: masking boolean tensor, same shape as tensor
            name: name for output tensor
        Returns:
            Same shape as val, where some elements are very small (exponentially zero)
        """
        VERY_BIG_NUMBER = 1e30
        VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER
        VERY_SMALL_NUMBER = 1e-30
        VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
        if name is None:
            name = "exp_mask"
        return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name=name)