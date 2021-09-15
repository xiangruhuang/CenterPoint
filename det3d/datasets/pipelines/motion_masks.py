from ..registry import PIPELINES

@PIPELINES.register_module
class AssignMotionLabel(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, res, info):
        import ipdb; ipdb.set_trace()
        return res, info
