class NormalizeFeatures(object):
    r"""Row-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.x = data.x / data.x.sum(1, keepdims=True).clamp(min_v=1)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
