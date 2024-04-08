from closure_base import ClosureBase
import cupy as cp


class ClosureFoundationPose(ClosureBase):

    def __init__(
        self, 
        pred_Rs: cp.ndarray, # (M, 3, 3)
        pred_ts: cp.ndarray, # (M, 3)
        pred_scores: cp.ndarray, # (M, )
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pred_Rs = pred_Rs
        self.pred_ts = pred_ts
        self.pred_scores = pred_scores
        