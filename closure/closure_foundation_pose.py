from closure_base import ClosureBase
import cupy as cp

import nonconformity_funcs as F


class ClosureFoundationPose(ClosureBase):

    def __init__(
        self,
        pred_Rs: cp.ndarray,  # (M, 3, 3)
        pred_ts: cp.ndarray,  # (M, 3)
        pred_scores: cp.ndarray,  # (M, )
        nonconformity_func_name: str,
        **kwargs
    ):
        self.pred_Rs = pred_Rs
        self.pred_ts = pred_ts
        self.pred_scores = pred_scores
        assert nonconformity_func_name in F.__dict__.keys()
        self.nonconformity_func = getattr(F, nonconformity_func_name)

        super().__init__(**kwargs)

    def nonconformity_func(
        self,
        center_Rs: cp.ndarray,  # (K, 3, 3)
        center_ts: cp.ndarray,  # (K, 3)
        pred_Rs: cp.ndarray,  # (M, 3, 3)
        pred_ts: cp.ndarray,  # (M, 3)
        pred_scores: cp.ndarray,  # (M, )
    ) -> cp.ndarray: ...
