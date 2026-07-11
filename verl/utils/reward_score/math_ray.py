

from .openmathinst_utils import extract_answer, math_equal

import ray
from ray.exceptions import GetTimeoutError
from math_verify import verify, parse
from typing import Union

# @ray.remote
def math_equal_ray(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    timeout: float = 3.0,
    check_antlr_version: bool = True
) -> bool:
    return math_equal(prediction, reference, include_percentage, tolerance, timeout, check_antlr_version)

# @ray.remote
def verify_ray(
    gold, 
    target, 
    float_rounding: int=6,
    numeric_precision: int=15,
    strict: bool=True,
    timeout_seconds: int=3
) -> bool:
    return verify(gold, target, float_rounding, numeric_precision, strict, timeout_seconds)

@ray.remote(num_gpus=0)
def compute_score(solution_str, ground_truth) -> float:

    omi_pred = None
    omi_correct = False
    mathv_pred = None
    mathv_correct = False

    try:
        omi_pred = extract_answer(solution_str, extract_from_boxed=True)
        # omi_correct_ref = math_equal_ray.remote(omi_pred, ground_truth, check_antlr_version=False)
        # omi_correct = ray.get(omi_correct_ref, timeout=3.0)
        omi_correct = math_equal(omi_pred, ground_truth, check_antlr_version=False, timeout=3.0, tolerance=1e-4)
    except GetTimeoutError as e:
        # ray.cancel(omi_correct_ref, force=True)
        omi_correct = False
    except Exception:
        omi_correct = False

    # math
    try:
        mathv_pred = parse(solution_str)
        # mathv_correct_ref = verify_ray.remote(parse(f"\\boxed{{${ground_truth}$}}"), mathv_pred)
        # mathv_correct = ray.get(mathv_correct_ref, timeout=3.0)
        mathv_correct = verify(parse(f"\\boxed{{${ground_truth}$}}"), mathv_pred, timeout_seconds=3, float_rounding=6, numeric_precision=15, strict=True)
    except GetTimeoutError as e:
        # ray.cancel(mathv_correct_ref, force=True)
        mathv_correct = False
    except Exception:
        mathv_correct = False

    acc = (omi_correct or mathv_correct)
    
    return float(acc)


@ray.remote
def compute_scores_chunk(sol_list, gt_list):
    out = []
    for s, g in zip(sol_list, gt_list):
        out.append(compute_score(s, g))
    return out


