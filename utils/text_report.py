"""Rule-based textual explanation for uncertainty-aware segmentation.

The text intentionally avoids making a clinical diagnosis. It reports model
reliability signals such as high uncertainty, possible missed lesion-like regions,
or prediction components that do not look well-supported by the training data.
"""
from __future__ import annotations

from typing import Dict, List


def risk_level(score: float, low: float = 0.33, high: float = 0.66) -> str:
    if score >= high:
        return "high"
    if score >= low:
        return "medium"
    return "low"


def failure_label_from_metrics(row: Dict[str, float]) -> str:
    dice = float(row.get("dice", 1.0))
    tp = int(row.get("tp_pixels", 1))
    fp = int(row.get("fp_pixels", 0))
    fn = int(row.get("fn_pixels", 0))
    missed = int(row.get("missed_component_count", 0))
    fp_comp = int(row.get("fp_component_count", 0))
    if tp == 0 and fp > 0 and fn > 0:
        return "complete_miss_with_false_positive"
    if tp == 0 and fn > 0:
        return "complete_miss"
    if dice < 0.50:
        return "severe_low_overlap"
    if missed > 0 and fp_comp > 0:
        return "component_miss_with_fp"
    if missed > 0:
        return "missed_lesion_component"
    if fp_comp > 0 and fp > fn:
        return "false_positive_components"
    return "acceptable_or_minor_error"


def build_text_explanation(row: Dict[str, float], language: str = "vi") -> str:
    """Generate a short case-level textual warning.

    Args:
        row: flat dict of metrics/uncertainty scores for one case.
        language: "vi" or "en".
    """
    score = float(row.get("case_uncertainty_score", 0.0))
    level = risk_level(score)
    label = str(row.get("failure_subtype", row.get("failure_label", "unknown")))

    dice = float(row.get("dice", -1.0))
    entropy = float(row.get("entropy_p95", 0.0))
    tta_var = float(row.get("tta_var_p95", 0.0))
    img_dist = float(row.get("train_image_distance", 0.0))
    comp_dist = float(row.get("pred_component_train_distance", 0.0))
    pred_comps = int(row.get("pred_component_count", 0))
    gt_comps = int(row.get("gt_component_count", 0))
    missed = int(row.get("missed_component_count", 0))
    fp_comp = int(row.get("fp_component_count", 0))
    tp = int(row.get("tp_pixels", 0))
    fp = int(row.get("fp_pixels", 0))
    fn = int(row.get("fn_pixels", 0))
    prob_gt_max = float(row.get("prob_in_gt_max", -1.0))
    prob_fp_max = float(row.get("prob_in_fp_max", -1.0))

    if language.lower().startswith("en"):
        parts: List[str] = []
        parts.append(f"Reliability warning: {level.upper()} uncertainty (score={score:.3f}).")
        if dice >= 0:
            parts.append(f"The current mask has Dice={dice:.3f} on this labeled evaluation case.")
        if label == "complete_miss_with_false_positive":
            parts.append("The prediction does not overlap the annotated lesion while still producing a positive component elsewhere; this suggests a complete localization failure with false positive prediction.")
        elif missed > 0:
            parts.append(f"There are {missed} annotated lesion component(s) not matched by the prediction.")
        if fp_comp > 0:
            parts.append(f"There are {fp_comp} predicted component(s) without overlap with the annotation.")
        if entropy > 0.55:
            parts.append("High probability entropy indicates ambiguous pixels near the decision boundary.")
        if tta_var > 0.01:
            parts.append("The prediction is unstable under test-time augmentation, suggesting model uncertainty.")
        if img_dist > 2.5:
            parts.append("The case appears relatively far from the training image distribution.")
        if comp_dist > 2.5:
            parts.append("At least one predicted component is not well supported by the training lesion-component distribution.")
        if prob_gt_max >= 0 and prob_gt_max < 0.30 and fn > 0:
            parts.append("The model assigns low probability inside the annotated lesion, suggesting a possible missed lesion-like region.")
        if prob_fp_max > 0.80 and fp > 0:
            parts.append("The model is highly confident in a region outside the annotation; this may require careful review.")
        parts.append("This text is an automatic model-reliability explanation, not a clinical diagnosis.")
        return " ".join(parts)

    parts = []
    parts.append(f"Cảnh báo độ tin cậy: mức {level.upper()} (score={score:.3f}).")
    if dice >= 0:
        parts.append(f"Trên case có nhãn này, mask hiện tại đạt Dice={dice:.3f}.")
    if label == "complete_miss_with_false_positive":
        parts.append("Dự đoán không chồng lên vùng lesion được gán nhãn nhưng vẫn tạo vùng dương tính ở nơi khác; đây là dấu hiệu fail định vị hoàn toàn kèm false positive.")
    elif missed > 0:
        parts.append(f"Có {missed} vùng lesion trong nhãn chưa được prediction bắt trúng.")
    if fp_comp > 0:
        parts.append(f"Có {fp_comp} vùng prediction không overlap với nhãn, có nguy cơ là false positive.")
    if entropy > 0.55:
        parts.append("Entropy cao cho thấy nhiều pixel gần ranh giới quyết định, model đang mơ hồ ở vùng biên hoặc vùng nghi ngờ.")
    if tta_var > 0.01:
        parts.append("Prediction thay đổi khi dùng test-time augmentation, cho thấy model không ổn định trên case này.")
    if img_dist > 2.5:
        parts.append("Ảnh này tương đối xa phân bố ảnh training, có thể là dạng case model ít gặp.")
    if comp_dist > 2.5:
        parts.append("Ít nhất một vùng prediction không giống các lesion component đã thấy trong training, nên độ tin cậy thấp.")
    if prob_gt_max >= 0 and prob_gt_max < 0.30 and fn > 0:
        parts.append("Model gán xác suất thấp trong vùng lesion được annotate, gợi ý nguy cơ bỏ sót lesion.")
    if prob_fp_max > 0.80 and fp > 0:
        parts.append("Model rất tự tin ở vùng ngoài annotation; vùng này cần được kiểm tra kỹ vì có thể là false positive tự tin cao.")
    parts.append("Đây là giải thích tự động về độ tin cậy của model, không phải kết luận chẩn đoán thay bác sĩ.")
    return " ".join(parts)
