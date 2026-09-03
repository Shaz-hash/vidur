# utils/deepcopy_safe.py
class DeepcopySafeMixin:
    DROP_ATTRS: tuple[str, ...] = ()
    def REINIT_AFTER_RESTORE(self):  # hook for derived classes
        pass
    def __getstate__(self):
        d = self.__dict__.copy()
        for name in self.DROP_ATTRS:
            if name in d:
                d[name] = None
        return d
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.REINIT_AFTER_RESTORE()
