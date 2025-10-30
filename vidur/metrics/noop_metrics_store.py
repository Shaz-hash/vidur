# metrics/noop_cluster_metrics_store.py
class NoOpClusterMetricsStore:
    def __getattr__(self, _):
        def _noop(*args, **kwargs):
            return None
        return _noop
