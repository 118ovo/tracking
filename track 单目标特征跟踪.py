import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from ultralytics import YOLO

# =========================================================
# File settings / 文件设置
# =========================================================
MODEL_FILE = "yolov8n.pt"
VIDEO_FILE = "sample.mp4"
OUTPUT_FILE = "result.mp4"

# =========================================================
# Target initialization / 目标初始化
# =========================================================
# Local calibration mode / 本地标定模式
# True  -> manually select the target in the first frame / 第一帧手动框选
# False -> use the hardcoded INITIAL_BOX for final submission / 提交前使用硬编码框
USE_MANUAL_SELECTION = False

# Replace this after your local calibration / 用你的本地标定结果替换这里
# Format: (x1, y1, x2, y2)
INITIAL_BOX = (1011, 478, 1240, 1037)

# =========================================================
# Detection parameters / 检测参数
# =========================================================
CONF_THRES = 0.22
IOU_THRES = 0.45
IMG_SIZE = 640

# =========================================================
# Search and tracking parameters / 搜索与跟踪参数
# =========================================================
SEARCH_EXPAND_RATIO = 0.45
MIN_LOCAL_IOU = 0.03
MIN_LOCAL_DIST_SCORE = 0.72

MAX_LOST_FRAMES = 25
STOP_DRIFT_AFTER = 10
REACQUIRE_CONFIRM_FRAMES = 3
PROFILE_UPDATE_ALPHA = 0.04

# =========================================================
# Score thresholds / 分数阈值
# =========================================================
NORMAL_ACCEPT_THRES = 0.46
AMBIGUOUS_GAP = 0.08

# Re-identification feature gates / 重识别特征门槛
UPPER_REID_THRES = 0.72
LOWER_REID_THRES = 0.66
HEAD_REID_THRES = 0.45
GLOBAL_REID_THRES = 0.68
APPEARANCE_REID_THRES = 0.70

# =========================================================
# Normal mode weights / 正常模式权重
# =========================================================
W_UPPER = 0.28
W_LOWER = 0.24
W_HEAD = 0.10
W_GLOBAL = 0.16
W_SHAPE = 0.08
W_MOTION = 0.08
W_IOU = 0.06

# =========================================================
# Occlusion mode weights / 遮挡模式权重
# =========================================================
W_UPPER_OCC = 0.30
W_LOWER_OCC = 0.28
W_HEAD_OCC = 0.12
W_GLOBAL_OCC = 0.18
W_SHAPE_OCC = 0.07
W_MOTION_OCC = 0.03
W_IOU_OCC = 0.02


# =========================================================
# Data structure / 数据结构
# =========================================================
@dataclass
class TargetProfile:
    upper_hist: np.ndarray | None
    lower_hist: np.ndarray | None
    head_hist: np.ndarray | None
    global_hist: np.ndarray | None
    aspect_ratio: float
    area_ratio: float


# =========================================================
# Utility functions / 工具函数
# =========================================================
def compute_iou(box1, box2):
    """Compute IoU for two xyxy boxes / 计算两个 xyxy 框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-6
    return inter / union


def box_center(box):
    """Get center of a box / 获取框中心"""
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def box_size(box):
    """Get width and height / 获取宽高"""
    return max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])


def normalized_center_score(box1, box2, frame_w, frame_h):
    """Distance-based score in [0,1] / 基于中心距离的分数"""
    c1x, c1y = box_center(box1)
    c2x, c2y = box_center(box2)
    dist = np.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2)
    diag = np.sqrt(frame_w ** 2 + frame_h ** 2) + 1e-6
    return 1.0 - min(dist / diag, 1.0)


def shape_similarity(box1, box2, frame_area):
    """Compare shape and size / 比较形状和尺寸"""
    w1, h1 = box_size(box1)
    w2, h2 = box_size(box2)

    ar1 = w1 / h1
    ar2 = w2 / h2
    ar_score = min(ar1, ar2) / max(ar1, ar2)

    a1 = (w1 * h1) / frame_area
    a2 = (w2 * h2) / frame_area
    area_score = min(a1, a2) / max(a1, a2 + 1e-6)

    return 0.5 * ar_score + 0.5 * area_score


def crop_box(frame, box):
    """Crop ROI from frame / 从图像中裁剪目标区域"""
    h, w = frame.shape[:2]
    x1 = max(0, int(round(box[0])))
    y1 = max(0, int(round(box[1])))
    x2 = min(w - 1, int(round(box[2])))
    y2 = min(h - 1, int(round(box[3])))

    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def compute_hist(image):
    """Compute normalized HSV histogram / 计算归一化 HSV 直方图"""
    if image is None or image.size == 0:
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [20], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [20], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [20], [0, 256]).flatten()

    hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm == 0:
        return None
    return hist / norm


def hist_similarity(hist1, hist2):
    """Cosine similarity of two histograms / 两个直方图的余弦相似度"""
    if hist1 is None or hist2 is None:
        return 0.0
    score = float(np.dot(hist1, hist2))
    return max(0.0, min(score, 1.0))


def split_body_regions(frame, box):
    """
    Split target box into head, upper, lower, and global regions
    / 将目标框分成头部、上半身、下半身和整体区域
    """
    roi = crop_box(frame, box)
    if roi is None or roi.size == 0:
        return None, None, None, None

    h, w = roi.shape[:2]
    if h < 10 or w < 5:
        return None, None, None, roi

    head_y2 = max(1, int(h * 0.18))
    upper_y1 = int(h * 0.12)
    upper_y2 = max(upper_y1 + 1, int(h * 0.58))
    lower_y1 = int(h * 0.55)

    head = roi[0:head_y2, :]
    upper = roi[upper_y1:upper_y2, :]
    lower = roi[lower_y1:h, :]
    global_roi = roi

    return head, upper, lower, global_roi


def extract_profile(frame, box):
    """Extract target profile / 提取目标画像"""
    head, upper, lower, global_roi = split_body_regions(frame, box)
    h, w = frame.shape[:2]
    bw, bh = box_size(box)
    frame_area = float(h * w)

    return TargetProfile(
        upper_hist=compute_hist(upper),
        lower_hist=compute_hist(lower),
        head_hist=compute_hist(head),
        global_hist=compute_hist(global_roi),
        aspect_ratio=bw / bh,
        area_ratio=(bw * bh) / frame_area,
    )


def blend_profile(old_profile, new_profile, alpha=0.04):
    """EMA update for target profile / 用指数滑动平均更新目标画像"""
    if old_profile is None:
        return new_profile

    def blend_hist(h1, h2):
        if h1 is None:
            return h2
        if h2 is None:
            return h1
        h = (1 - alpha) * h1 + alpha * h2
        norm = np.linalg.norm(h)
        if norm == 0:
            return h1
        return h / norm

    return TargetProfile(
        upper_hist=blend_hist(old_profile.upper_hist, new_profile.upper_hist),
        lower_hist=blend_hist(old_profile.lower_hist, new_profile.lower_hist),
        head_hist=blend_hist(old_profile.head_hist, new_profile.head_hist),
        global_hist=blend_hist(old_profile.global_hist, new_profile.global_hist),
        aspect_ratio=(1 - alpha) * old_profile.aspect_ratio + alpha * new_profile.aspect_ratio,
        area_ratio=(1 - alpha) * old_profile.area_ratio + alpha * new_profile.area_ratio,
    )


def profile_similarity(profile, frame, box):
    """Compare candidate with target profile / 比较候选人与目标画像"""
    cand = extract_profile(frame, box)

    upper_score = hist_similarity(profile.upper_hist, cand.upper_hist)
    lower_score = hist_similarity(profile.lower_hist, cand.lower_hist)
    head_score = hist_similarity(profile.head_hist, cand.head_hist)
    global_score = hist_similarity(profile.global_hist, cand.global_hist)

    ar_score = min(profile.aspect_ratio, cand.aspect_ratio) / max(profile.aspect_ratio, cand.aspect_ratio + 1e-6)
    area_score = min(profile.area_ratio, cand.area_ratio) / max(profile.area_ratio, cand.area_ratio + 1e-6)
    shape_score = 0.5 * ar_score + 0.5 * area_score

    return {
        "upper": upper_score,
        "lower": lower_score,
        "head": head_score,
        "global": global_score,
        "shape": shape_score,
        "candidate_profile": cand,
    }


def appearance_only_score(candidate_scores):
    """
    Appearance-only score for re-identification
    / 用于重识别的纯外观分数
    """
    return (
        0.35 * candidate_scores["upper"] +
        0.30 * candidate_scores["lower"] +
        0.15 * candidate_scores["head"] +
        0.20 * candidate_scores["global"]
    )


def passes_reid_gate(candidate_scores):
    """
    Strict feature gate for re-identification
    / 严格的重识别特征门槛
    """
    app_score = appearance_only_score(candidate_scores)

    ok = (
        candidate_scores["upper"] >= UPPER_REID_THRES and
        candidate_scores["lower"] >= LOWER_REID_THRES and
        candidate_scores["head"] >= HEAD_REID_THRES and
        candidate_scores["global"] >= GLOBAL_REID_THRES and
        app_score >= APPEARANCE_REID_THRES
    )
    return ok, app_score


def expand_box(box, frame_w, frame_h, ratio=0.45):
    """Expand box for local search / 扩大搜索框用于局部搜索"""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    dx = w * ratio
    dy = h * ratio
    return [
        max(0.0, x1 - dx),
        max(0.0, y1 - dy),
        min(float(frame_w - 1), x2 + dx),
        min(float(frame_h - 1), y2 + dy),
    ]


def predict_next_box(last_box, velocity, frame_w, frame_h):
    """Predict next box with constant velocity / 用恒速模型预测下一帧框"""
    x1, y1, x2, y2 = last_box
    vx, vy = velocity

    nx1 = x1 + vx
    ny1 = y1 + vy
    nx2 = x2 + vx
    ny2 = y2 + vy

    bw = x2 - x1
    bh = y2 - y1

    nx1 = max(0.0, min(nx1, frame_w - bw - 1))
    ny1 = max(0.0, min(ny1, frame_h - bh - 1))
    nx2 = nx1 + bw
    ny2 = ny1 + bh
    return [nx1, ny1, nx2, ny2]


def detect_persons(model, frame):
    """Detect person class only / 仅检测 person 类"""
    result = model.predict(
        source=frame,
        classes=[0],
        conf=CONF_THRES,
        iou=IOU_THRES,
        imgsz=IMG_SIZE,
        verbose=False,
        device="cpu"
    )[0]

    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = box.tolist()
        detections.append({
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "conf": float(conf),
        })
    return detections


def select_initial_box(frame):
    """Manual ROI selection on the first frame / 在第一帧手动框选目标"""
    roi = cv2.selectROI("Select Target", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Target")
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return None
    return [float(x), float(y), float(x + w), float(y + h)]


def choose_candidates_near_prediction(detections, pred_box, frame_w, frame_h):
    """Keep candidates near predicted region / 保留预测区域附近的候选人"""
    search_box = expand_box(pred_box, frame_w, frame_h, SEARCH_EXPAND_RATIO)
    selected = []

    for det in detections:
        box = det["box"]
        overlap = compute_iou(search_box, box)
        dist_score = normalized_center_score(pred_box, box, frame_w, frame_h)
        if overlap > MIN_LOCAL_IOU or dist_score > MIN_LOCAL_DIST_SCORE:
            selected.append(det)

    if len(selected) == 0:
        return detections
    return selected


def score_candidate(candidate_box, pred_box, target_profile, frame, frame_area, occluded=False):
    """Score one candidate person / 对候选人进行综合打分"""
    prof = profile_similarity(target_profile, frame, candidate_box)

    motion_score = normalized_center_score(pred_box, candidate_box, frame.shape[1], frame.shape[0])
    iou_score = compute_iou(pred_box, candidate_box)
    shape_score = shape_similarity(pred_box, candidate_box, frame_area)

    if occluded:
        total = (
            W_UPPER_OCC * prof["upper"] +
            W_LOWER_OCC * prof["lower"] +
            W_HEAD_OCC * prof["head"] +
            W_GLOBAL_OCC * prof["global"] +
            W_SHAPE_OCC * min(shape_score, prof["shape"]) +
            W_MOTION_OCC * motion_score +
            W_IOU_OCC * iou_score
        )
    else:
        total = (
            W_UPPER * prof["upper"] +
            W_LOWER * prof["lower"] +
            W_HEAD * prof["head"] +
            W_GLOBAL * prof["global"] +
            W_SHAPE * min(shape_score, prof["shape"]) +
            W_MOTION * motion_score +
            W_IOU * iou_score
        )

    return {
        "score": float(total),
        "upper": prof["upper"],
        "lower": prof["lower"],
        "head": prof["head"],
        "global": prof["global"],
        "shape": prof["shape"],
        "candidate_profile": prof["candidate_profile"],
    }


def detect_ambiguity(scored_candidates, frame_w, frame_h):
    """
    Detect ambiguous crowding state
    / 检测候选人是否过于接近、容易混淆
    """
    if len(scored_candidates) < 2:
        return False

    scored_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)
    best = scored_candidates[0]
    second = scored_candidates[1]

    gap_small = (best["score"] - second["score"]) < AMBIGUOUS_GAP
    overlap = compute_iou(best["box"], second["box"]) > 0.12
    close_pos = normalized_center_score(best["box"], second["box"], frame_w, frame_h) > 0.90

    return gap_small and (overlap or close_pos)


def draw_target(frame, box, label="target", detail=""):
    """Draw target box and optional detail / 绘制目标框和附加说明"""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.putText(
        frame,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2
    )

    if detail:
        cv2.putText(
            frame,
            detail,
            (x1, max(55, y1 - 35)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )

    return frame


# =========================================================
# Main function / 主函数
# =========================================================
def main():
    """Run profile-based re-identification tracking / 运行基于画像重识别的跟踪"""
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / MODEL_FILE
    video_path = base_dir / VIDEO_FILE
    output_path = base_dir / OUTPUT_FILE

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print("Loading YOLOv8n model...")
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = float(width * height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        writer.release()
        raise RuntimeError("Cannot read the first frame.")

    # -----------------------------------------------------
    # Initialize target / 初始化目标
    # -----------------------------------------------------
    if USE_MANUAL_SELECTION:
        print("Please draw the target box on the first frame, then press ENTER.")
        init_box = select_initial_box(first_frame)
        if init_box is None:
            cap.release()
            writer.release()
            raise RuntimeError("No target selected on the first frame.")
        print("Suggested INITIAL_BOX =", tuple(int(v) for v in init_box))
        print("Please copy this box into the code and set USE_MANUAL_SELECTION = False before submission.")
    else:
        if INITIAL_BOX is None:
            cap.release()
            writer.release()
            raise RuntimeError("Please set INITIAL_BOX before submission.")
        init_box = list(map(float, INITIAL_BOX))

    first_detections = detect_persons(model, first_frame)
    if len(first_detections) == 0:
        cap.release()
        writer.release()
        raise RuntimeError("No person detected in the first frame.")

    best_first = None
    best_first_iou = -1.0
    for det in first_detections:
        score = compute_iou(init_box, det["box"])
        if score > best_first_iou:
            best_first_iou = score
            best_first = det

    if best_first is None:
        cap.release()
        writer.release()
        raise RuntimeError("Failed to match the target in the first frame.")

    target_box = best_first["box"]
    target_profile = extract_profile(first_frame, target_box)

    # State variables / 状态变量
    velocity = np.array([0.0, 0.0], dtype=np.float32)
    lost_count = 0
    occluded = False
    reacquire_box = None
    reacquire_count = 0
    tracking_state = "TRACKING"
    display_score = 0.0
    detail_text = ""

    first_vis = draw_target(first_frame.copy(), target_box, label="ID:1 TRACKING", detail="")
    writer.write(first_vis)

    print("Start tracking...")

    frame_idx = 1

    # -----------------------------------------------------
    # Tracking loop / 跟踪循环
    # -----------------------------------------------------
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        pred_box = predict_next_box(target_box, velocity, width, height)

        # Stop drifting after too many lost frames / 长时间丢失后停止漂移
        if lost_count > STOP_DRIFT_AFTER:
            pred_box = target_box
            velocity[:] = 0.0

        detections = detect_persons(model, frame)

        if len(detections) > 0:
            nearby = choose_candidates_near_prediction(detections, pred_box, width, height)

            scored = []
            for det in nearby:
                s = score_candidate(det["box"], pred_box, target_profile, frame, frame_area, occluded=occluded)
                scored.append({
                    "box": det["box"],
                    "conf": det["conf"],
                    "score": s["score"],
                    "upper": s["upper"],
                    "lower": s["lower"],
                    "head": s["head"],
                    "global": s["global"],
                    "shape": s["shape"],
                    "candidate_profile": s["candidate_profile"],
                })

            scored = sorted(scored, key=lambda x: x["score"], reverse=True)
            is_ambiguous = detect_ambiguity(scored, width, height)

            if not occluded:
                # Normal mode / 正常模式
                if len(scored) > 0 and (not is_ambiguous) and scored[0]["score"] >= NORMAL_ACCEPT_THRES:
                    best = scored[0]
                    new_box = best["box"]

                    old_center = np.array(box_center(target_box), dtype=np.float32)
                    new_center = np.array(box_center(new_box), dtype=np.float32)
                    velocity = 0.70 * velocity + 0.30 * (new_center - old_center)

                    target_box = new_box
                    target_profile = blend_profile(target_profile, best["candidate_profile"], PROFILE_UPDATE_ALPHA)
                    lost_count = 0
                    reacquire_box = None
                    reacquire_count = 0
                    tracking_state = "TRACKING"
                    display_score = best["score"]
                    detail_text = (
                        f"U:{best['upper']:.2f} "
                        f"L:{best['lower']:.2f} "
                        f"H:{best['head']:.2f} "
                        f"G:{best['global']:.2f}"
                    )
                else:
                    # Enter occlusion mode / 进入遮挡模式
                    occluded = True
                    target_box = pred_box
                    lost_count += 1
                    reacquire_box = None
                    reacquire_count = 0
                    tracking_state = "OCCLUDED"
                    display_score = 0.0
                    detail_text = ""
            else:
                # Occlusion mode / 遮挡模式
                passed_candidates = []

                for cand in scored:
                    ok_reid, app_score = passes_reid_gate(cand)
                    cand["app_score"] = app_score
                    cand["reid_ok"] = ok_reid
                    if ok_reid:
                        passed_candidates.append(cand)

                passed_candidates = sorted(
                    passed_candidates,
                    key=lambda x: x["app_score"],
                    reverse=True
                )

                if len(passed_candidates) > 0:
                    cand = passed_candidates[0]
                    candidate_box = cand["box"]

                    if reacquire_box is None:
                        reacquire_box = candidate_box
                        reacquire_count = 1
                    else:
                        if compute_iou(reacquire_box, candidate_box) > 0.50:
                            reacquire_box = candidate_box
                            reacquire_count += 1
                        else:
                            reacquire_box = candidate_box
                            reacquire_count = 1

                    if reacquire_count >= REACQUIRE_CONFIRM_FRAMES:
                        old_center = np.array(box_center(target_box), dtype=np.float32)
                        new_center = np.array(box_center(candidate_box), dtype=np.float32)
                        velocity = 0.50 * velocity + 0.50 * (new_center - old_center)

                        target_box = candidate_box
                        target_profile = blend_profile(
                            target_profile,
                            cand["candidate_profile"],
                            PROFILE_UPDATE_ALPHA * 0.5
                        )
                        lost_count = 0
                        occluded = False
                        tracking_state = "REID_BY_FEATURE"
                        display_score = cand["app_score"]
                        detail_text = (
                            f"U:{cand['upper']:.2f} "
                            f"L:{cand['lower']:.2f} "
                            f"H:{cand['head']:.2f} "
                            f"G:{cand['global']:.2f}"
                        )
                        reacquire_box = None
                        reacquire_count = 0
                    else:
                        target_box = pred_box
                        lost_count += 1
                        tracking_state = "OCCLUDED"
                        display_score = 0.0
                        detail_text = ""
                else:
                    target_box = pred_box
                    lost_count += 1
                    tracking_state = "OCCLUDED"
                    display_score = 0.0
                    detail_text = ""
        else:
            occluded = True
            target_box = pred_box
            lost_count += 1
            tracking_state = "OCCLUDED"
            display_score = 0.0
            detail_text = ""

        lost_count = min(lost_count, MAX_LOST_FRAMES)

        if tracking_state == "OCCLUDED":
            label = f"ID:1 OCCLUDED L:{lost_count}"
        elif tracking_state == "REID_BY_FEATURE":
            label = f"ID:1 REID_BY_FEATURE A:{display_score:.2f}"
        else:
            label = f"ID:1 TRACKING A:{display_score:.2f}"

        vis = draw_target(frame.copy(), target_box, label=label, detail=detail_text)
        writer.write(vis)

    cap.release()
    writer.release()
    print(f"Done. Result saved to: {output_path}")


if __name__ == "__main__":
    main()