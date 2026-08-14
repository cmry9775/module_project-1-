import numpy as np

LO = 0.6931471805599453
HI = 9.907678757154596

def score_single(value: int) -> int:
    log_val = np.log1p(value)
    score = round(100 - (log_val - LO) / (HI - LO) * 99)
    return int(score)