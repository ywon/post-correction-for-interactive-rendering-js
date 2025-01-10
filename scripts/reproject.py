import tensorflow as tf

_module = tf.load_op_library('./ops/reproject.so')

Reproject = _module.reproject
ReprojectVariance = _module.reproject_variance
CalShrinkage = _module.cal_shrinkage
AvgShrinkage = _module.avg_shrinkage
BoxFilter = _module.box_filter
WeightAvg = _module.weight_avg
OutlierRemoval = _module.outlier_removal
cuda_synchronize = _module.cuda_synchronize

def luminance(rgb):
    return rgb[...,0:1] * 0.2126 + rgb[...,1:2] * 0.7152 + rgb[...,2:3] * 0.0722

# a * t2 + (1 - a) * t1
def lerp(t1, t2, alpha):
    return t1 + alpha * (t2 - t1)

def reproject(input, current, mvec, pnFwidth, linearZ, prev_linearZ, normal, prev_normal, ALPHA=0.2, MOMENTS_ALPHA=0.2):
    height, width, num_channels = input.shape[1:4]
    success, reproj_output = Reproject(mvec, input, linearZ, prev_linearZ, normal, prev_normal, pnFwidth, height, width, num_channels)
    history, historylen, moments_prev = tf.split(reproj_output, [3, 1, 2], axis=-1)

    accumlen = tf.minimum(32, tf.where(success, historylen + 1, 1))
    inv_len = tf.math.reciprocal(accumlen)
    alpha = tf.maximum(ALPHA, inv_len)
    alpha_moments = tf.maximum(MOMENTS_ALPHA, inv_len)
    accum = lerp(history, current, alpha)

    lum = luminance(current)
    moments = tf.concat([lum, lum * lum], axis=-1)
    isLargerThanOne = accumlen > 1.0
    new_moments = lerp(moments_prev, moments, alpha_moments)
    moments = tf.where(isLargerThanOne, new_moments, moments)
    variance = tf.where(isLargerThanOne, tf.maximum(0, moments[...,1:2] - moments[...,0:1] * moments[...,0:1]), 0)

    return success, history, historylen, accum, accumlen, variance, moments, moments_prev


class Buffer:
    def __init__(self, u) -> None:
        # Initialize
        self.accum = u.current_demodul
        self.accumlen = tf.ones_like(u.ones1)
        self.history = tf.zeros_like(u.zeros3)
        self.historylen = tf.zeros_like(u.zeros1)
        self.moments = tf.zeros_like(u.zeros2) # (1st, 2nd) moments
        self.moments_prev = tf.zeros_like(u.zeros2)
        self.variance = tf.zeros_like(u.zeros1)
        self.visibility = tf.zeros_like(u.zeros1)
        self.visibility_prev = tf.zeros_like(u.zeros1)

    def get_reproj_input(self, ignore_prev=False):
        if ignore_prev:
            return [self.history, self.historylen, self.moments_prev]
        else:
            return [self.accum, self.accumlen, self.moments]
    
    def update(self, history, historylen, accum=None, accumlen=None, variance=None, moments=None, moments_prev=None):
        # Update buffer
        self.history = history
        self.historylen = historylen
        if accum is not None: self.accum = accum
        if accumlen is not None: self.accumlen = accumlen
        if variance is not None: self.variance = variance
        if moments is not None: self.moments = moments
        if moments_prev is not None: self.moments_prev = moments_prev