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
# Assignment target / 老师指定目标
# =========================================================
ASSIGNMENT_INITIAL_BOX = (1011, 478, 1240, 1037)

# =========================================================
# Detection parameters / 检测参数
# =========================================================
CONF_THRES = 0.22
IOU_THRES = 0.45
IMG_SIZE = 640

# =========================================================
# Tracking parameters / 跟踪参数
# =========================================================
SEARCH_EXPAND_RATIO = 0.60
MIN_LOCAL_IOU = 0.01
MIN_LOCAL_DIST_SCORE = 0.65

MAX_LOST_FRAMES = 40
REACQUIRE_CONFIRM_FRAMES = 3
PROFILE_UPDATE_ALPHA = 0.03

TRAIL_SECONDS = 3.0

# Warm-up settings / Warm-up 设置
WARMUP_FRAMES = 3
WARMUP_ACCEPT_THRES = 0.50
WARMUP_PROFILE_ALPHA = 0.10

# =========================================================
# Score thresholds / 分数阈值
# =========================================================
NORMAL_ACCEPT_THRES = 0.42
AMBIGUOUS_GAP = 0.08

# Normal tracking feature gate / 正常跟踪特征门槛
TRACK_UPPER_THRES = 0.55
TRACK_LOWER_THRES = 0.45
TRACK_GLOBAL_THRES = 0.55
TRACK_APPEARANCE_THRES = 0.56

# Re-identification gates / 重识别门槛
UPPER_REID_THRES = 0.62
LOWER_REID_THRES = 0.58
HEAD_REID_THRES = 0.25
GLOBAL_REID_THRES = 0.60
APPEARANCE_REID_THRES = 0.64
REID_MARGIN_THRES = 0.04

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
# Lost / ReID mode weights / 丢失与重识别模式权重
# =========================================================
W_UPPER_OCC = 0.30
W_LOWER_OCC = 0.28
W_HEAD_OCC = 0.12
W_GLOBAL_OCC = 0.18
W_SHAPE_OCC = 0.07
W_MOTION_OCC = 0.03
W_IOU_OCC = 0.02


@dataclass
class TargetProfile:
    upper_hist: np.ndarray | None
    lower_hist: np.ndarray | None
    head_hist: np.ndarray | None
    global_hist: np.ndarray | None
    aspect_ratio: float
    area_ratio: float


# =========================================================
# Basic box functions / 基础框函数
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
    """Get box center / 获取框中心"""
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def box_size(box):
    """Get box size / 获取框尺寸"""
    return max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])


def normalized_center_score(box1, box2, frame_w, frame_h):
    """Distance score in [0,1] / 中心距离分数"""
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


# =========================================================
# Feature extraction / 特征提取
# =========================================================
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
    """Cosine similarity / 余弦相似度"""
    if hist1 is None or hist2 is None:
        return 0.0

    score = float(np.dot(hist1, hist2))
    return max(0.0, min(score, 1.0))


def split_body_regions(frame, box):
    """
    Split a person box into head, upper body, lower body, and global regions.
    将人物框分成头部、上半身、下半身和整体区域。
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


def blend_profile(old_profile, new_profile, alpha=0.03):
    """Update target profile by EMA / 用指数滑动平均更新目标画像"""
    def blend_hist(h1, h2):
        if h1 is None:
            return h2
        if h2 is None:
            return h1

        h = (1.0 - alpha) * h1 + alpha * h2
        norm = np.linalg.norm(h)

        if norm == 0:
            return h1

        return h / norm

    return TargetProfile(
        upper_hist=blend_hist(old_profile.upper_hist, new_profile.upper_hist),
        lower_hist=blend_hist(old_profile.lower_hist, new_profile.lower_hist),
        head_hist=blend_hist(old_profile.head_hist, new_profile.head_hist),
        global_hist=blend_hist(old_profile.global_hist, new_profile.global_hist),
        aspect_ratio=(1.0 - alpha) * old_profile.aspect_ratio + alpha * new_profile.aspect_ratio,
        area_ratio=(1.0 - alpha) * old_profile.area_ratio + alpha * new_profile.area_ratio,
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
    """Appearance-only score for re-identification / 重识别时的纯外观分数"""
    return (
        0.35 * candidate_scores["upper"] +
        0.30 * candidate_scores["lower"] +
        0.15 * candidate_scores["head"] +
        0.20 * candidate_scores["global"]
    )


def passes_tracking_gate(candidate_scores):
    """
    Normal tracking gate.
    正常跟踪门槛：如果不够像目标，就直接判定丢失，不把框切到别人身上。
    """
    app_score = appearance_only_score(candidate_scores)

    ok = (
        candidate_scores["upper"] >= TRACK_UPPER_THRES and
        candidate_scores["lower"] >= TRACK_LOWER_THRES and
        candidate_scores["global"] >= TRACK_GLOBAL_THRES and
        app_score >= TRACK_APPEARANCE_THRES
    )

    return ok, app_score


def passes_reid_gate(candidate_scores):
    """
    Strict feature gate for re-identification.
    严格的重识别门槛：只有足够像目标，才允许重新识别。
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


# =========================================================
# Detection and matching / 检测与匹配
# =========================================================
def detect_persons(model, frame):
    """Detect person class only / 只检测 person 类"""
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


def expand_box(box, frame_w, frame_h, ratio=0.60):
    """Expand a box for local search / 扩大搜索区域"""
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
    """Predict next box by constant velocity / 用恒速模型预测下一帧框"""
    x1, y1, x2, y2 = last_box
    vx, vy = velocity

    bw = x2 - x1
    bh = y2 - y1

    nx1 = x1 + vx
    ny1 = y1 + vy

    nx1 = max(0.0, min(nx1, frame_w - bw - 1))
    ny1 = max(0.0, min(ny1, frame_h - bh - 1))

    nx2 = nx1 + bw
    ny2 = ny1 + bh

    return [nx1, ny1, nx2, ny2]


def choose_candidates_near_prediction(detections, pred_box, frame_w, frame_h):
    """Keep candidates near predicted region / 只保留预测区域附近的候选人"""
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


def score_candidate(candidate_box, ref_box, target_profile, frame, frame_area, occluded=False):
    """Score one candidate person / 对候选人打分"""
    prof = profile_similarity(target_profile, frame, candidate_box)

    motion_score = normalized_center_score(ref_box, candidate_box, frame.shape[1], frame.shape[0])
    iou_score = compute_iou(ref_box, candidate_box)
    shape_score = shape_similarity(ref_box, candidate_box, frame_area)

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
    """Detect ambiguous candidates / 判断候选人是否过于接近或混淆"""
    if len(scored_candidates) < 2:
        return False

    scored_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)
    best = scored_candidates[0]
    second = scored_candidates[1]

    gap_small = (best["score"] - second["score"]) < AMBIGUOUS_GAP
    overlap = compute_iou(best["box"], second["box"]) > 0.12
    close_pos = normalized_center_score(best["box"], second["box"], frame_w, frame_h) > 0.90

    return gap_small and (overlap or close_pos)


# =========================================================
# Drawing functions / 绘图函数
# =========================================================
def add_center_to_trail(center_trail, box, frame_idx, max_trail_frames):
    """Add confirmed center point / 添加确认后的中心点"""
    cx, cy = box_center(box)
    center_trail.append((frame_idx, (int(cx), int(cy))))
    return prune_center_trail(center_trail, frame_idx, max_trail_frames)


def prune_center_trail(center_trail, current_frame_idx, max_trail_frames):
    """Keep only recent points / 只保留最近几秒轨迹点"""
    return [
        item for item in center_trail
        if current_frame_idx - item[0] <= max_trail_frames
    ]


def draw_center_trail(frame, center_trail):
    """Draw center trajectory / 绘制中心点轨迹"""
    if len(center_trail) == 0:
        return frame

    points = np.array([p for _, p in center_trail], dtype=np.int32)

    if len(points) >= 2:
        cv2.polylines(
            frame,
            [points],
            isClosed=False,
            color=(0, 255, 255),
            thickness=3
        )

    for p in points:
        cv2.circle(frame, tuple(p), 4, (0, 255, 255), -1)

    return frame


def draw_target(frame, box, label="target", detail=""):
    """Draw target box / 绘制目标框"""
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


def draw_lost_message(frame, lost_count):
    """Draw target lost message / 绘制目标丢失提示"""
    h, w = frame.shape[:2]

    cv2.putText(
        frame,
        "TARGET LOST",
        (int(w * 0.34), int(h * 0.45)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (0, 0, 255),
        4
    )

    cv2.putText(
        frame,
        f"Searching by appearance features... Lost frames: {lost_count}",
        (int(w * 0.23), int(h * 0.52)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    return frame


def draw_top_instruction(frame, line1, line2=""):
    """Draw instruction text at the top / 在顶部绘制交互提示"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cv2.putText(
        frame,
        line1,
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    if line2:
        cv2.putText(
            frame,
            line2,
            (25, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

    return frame


# =========================================================
# Target selection / 目标选择
# =========================================================
def match_initial_box_to_detection(init_box, detections):
    """Match initial box to the nearest YOLO detection / 将初始框匹配到 YOLO 检测框"""
    best_det = None
    best_iou = -1.0

    for det in detections:
        score = compute_iou(init_box, det["box"])
        if score > best_iou:
            best_iou = score
            best_det = det

    if best_det is None:
        return None

    return best_det["box"]


def select_target_on_current_frame(frame, detections):
    """
    Select target by drawing ROI on current frame.
    在当前帧手动画框选择目标。
    """
    display = frame.copy()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det["box"])
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            display,
            f"P{i}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    display = draw_top_instruction(
        display,
        "Draw a box around the target.",
        "ENTER: confirm ROI | ESC: cancel"
    )

    print("Draw a box around the target, then press ENTER to confirm.")
    roi = cv2.selectROI("Select Target", display, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Target")

    x, y, w, h = roi

    if w <= 0 or h <= 0:
        return None

    manual_box = [float(x), float(y), float(x + w), float(y + h)]
    selected_box = match_initial_box_to_detection(manual_box, detections)

    if selected_box is None:
        return manual_box

    print("Selected initial box:", tuple(int(v) for v in selected_box))
    return selected_box


def select_target_from_clear_frame(model, video_path):
    """
    Let user play the video and select a clear frame.
    允许用户播放视频，并在清晰帧选择目标。
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    paused = False
    frame_idx = -1
    current_frame = None
    current_detections = []

    print("Custom clear-frame mode controls:")
    print("SPACE - pause / play")
    print("S     - select target on current frame")
    print("ENTER - confirm ROI after drawing the box")
    print("Q     - quit")

    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            detections = detect_persons(model, frame)

            current_frame = frame.copy()
            current_detections = detections

        if current_frame is None:
            continue

        preview = current_frame.copy()

        for i, det in enumerate(current_detections):
            x1, y1, x2, y2 = map(int, det["box"])
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                preview,
                f"P{i}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        preview = draw_top_instruction(
            preview,
            "SPACE: pause/play | S: select target | Q: quit",
            "After pressing S: draw ROI, then press ENTER to confirm."
        )

        cv2.imshow("Preview", preview)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(" "):
            paused = not paused

        elif key == ord("s") or key == ord("S"):
            paused = True
            selected_box = select_target_on_current_frame(current_frame, current_detections)

            if selected_box is not None:
                cap.release()
                cv2.destroyWindow("Preview")
                return frame_idx, current_frame.copy(), selected_box

        elif key == ord("q") or key == ord("Q"):
            break

    cap.release()
    cv2.destroyWindow("Preview")
    raise RuntimeError("No target selected.")


def ask_mode():
    """Ask user for running mode / 询问运行模式"""
    print("Select mode:")

    print("1 - Assignment target ‘the specific person’")
    print("2 - Select target on the first frame")
    print("3 - Select a target from the specified frame; if the target is already visible at the start, begin tracking it immediately")

    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in ("1", "2", "3"):
            return choice
        print("Invalid input. Please enter 1, 2, or 3.")


# =========================================================
# Warm-up / Warm-up 阶段
# =========================================================
def warmup_profile(
    cap,
    writer,
    model,
    target_box,
    target_profile,
    last_confirmed_box,
    velocity,
    width,
    height,
    frame_area,
    center_trail,
    frame_idx,
    max_trail_frames,
):
    """
    Warm-up stage to improve the target profile.
    Warm-up 阶段，用于补充目标画像。
    """
    print(f"Start warm-up for {WARMUP_FRAMES} frames...")

    for i in range(WARMUP_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        ref_box = last_confirmed_box
        pred_box = predict_next_box(ref_box, velocity, width, height)
        detections = detect_persons(model, frame)

        matched = False
        detail = ""

        if len(detections) > 0:
            nearby = choose_candidates_near_prediction(detections, pred_box, width, height)

            scored = []

            for det in nearby:
                s = score_candidate(det["box"], pred_box, target_profile, frame, frame_area, occluded=False)
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

            if len(scored) > 0:
                ok_track, app_score = passes_tracking_gate(scored[0])
            else:
                ok_track, app_score = False, 0.0

            if len(scored) > 0 and scored[0]["score"] >= WARMUP_ACCEPT_THRES and ok_track:
                best = scored[0]
                new_box = best["box"]

                old_center = np.array(box_center(last_confirmed_box), dtype=np.float32)
                new_center = np.array(box_center(new_box), dtype=np.float32)
                velocity = 0.60 * velocity + 0.40 * (new_center - old_center)

                target_box = new_box
                last_confirmed_box = new_box.copy()
                target_profile = blend_profile(target_profile, best["candidate_profile"], WARMUP_PROFILE_ALPHA)
                matched = True

                detail = (
                    f"U:{best['upper']:.2f} "
                    f"L:{best['lower']:.2f} "
                    f"H:{best['head']:.2f} "
                    f"G:{best['global']:.2f}"
                )

        if matched:
            center_trail = add_center_to_trail(center_trail, last_confirmed_box, frame_idx, max_trail_frames)
        else:
            center_trail = prune_center_trail(center_trail, frame_idx, max_trail_frames)

        vis = frame.copy()
        vis = draw_target(vis, last_confirmed_box, label=f"ID:1 WARMUP {i + 1}/{WARMUP_FRAMES}", detail=detail)
        vis = draw_center_trail(vis, center_trail)
        writer.write(vis)

    print("Warm-up finished.")
    return target_box, target_profile, last_confirmed_box, velocity, center_trail, frame_idx


# =========================================================
# Main / 主函数
# =========================================================
def main():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / MODEL_FILE
    video_path = base_dir / VIDEO_FILE
    output_path = base_dir / OUTPUT_FILE

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    mode = ask_mode()

    print("Loading YOLOv8n model...")
    model = YOLO(str(model_path))

    info_cap = cv2.VideoCapture(str(video_path))

    if not info_cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = info_cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(info_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(info_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = float(width * height)
    max_trail_frames = int(fps * TRAIL_SECONDS)

    info_cap.release()

    start_with_confirmed_target = True

    if mode in ("1", "2"):
        cap_init = cv2.VideoCapture(str(video_path))
        ok, first_frame = cap_init.read()
        cap_init.release()

        if not ok:
            raise RuntimeError("Cannot read the first frame.")

        first_detections = detect_persons(model, first_frame)

        if len(first_detections) == 0:
            raise RuntimeError("No person detected in the first frame.")

        if mode == "1":
            init_box = list(map(float, ASSIGNMENT_INITIAL_BOX))
            target_box = match_initial_box_to_detection(init_box, first_detections)

            if target_box is None:
                raise RuntimeError("Failed to match the assignment target.")

            print("Mode 1 selected: assignment target.")

        else:
            target_box = select_target_on_current_frame(first_frame, first_detections)

            if target_box is None:
                raise RuntimeError("No target selected on the first frame.")

            print("Mode 2 selected: target selected on the first frame.")

        target_profile = extract_profile(first_frame, target_box)
        last_confirmed_box = target_box.copy()

    else:
        print("Mode 3 selected: select a clear frame first, then track from the beginning.")

        selected_frame_idx, selected_frame, selected_box = select_target_from_clear_frame(model, video_path)

        target_box = selected_box.copy()
        last_confirmed_box = selected_box.copy()
        target_profile = extract_profile(selected_frame, selected_box)

        start_with_confirmed_target = False

        print(f"Target profile was built from frame {selected_frame_idx}.")
        print("Now restarting the video from frame 0...")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    velocity = np.array([0.0, 0.0], dtype=np.float32)

    if start_with_confirmed_target:
        lost_count = 0
        occluded = False
        tracking_state = "TRACKING"
    else:
        lost_count = 1
        occluded = True
        tracking_state = "LOST"

    reacquire_box = None
    reacquire_count = 0
    display_score = 0.0
    detail_text = ""

    frame_idx = -1
    center_trail = []

    if start_with_confirmed_target:
        ok, first_frame = cap.read()
        if not ok:
            cap.release()
            writer.release()
            raise RuntimeError("Cannot read the first frame.")

        frame_idx += 1

        center_trail = add_center_to_trail(
            center_trail,
            last_confirmed_box,
            frame_idx,
            max_trail_frames
        )

        first_vis = first_frame.copy()
        first_vis = draw_target(first_vis, last_confirmed_box, label="ID:1 TRACKING", detail="")
        first_vis = draw_center_trail(first_vis, center_trail)
        writer.write(first_vis)

        target_box, target_profile, last_confirmed_box, velocity, center_trail, frame_idx = warmup_profile(
            cap,
            writer,
            model,
            target_box,
            target_profile,
            last_confirmed_box,
            velocity,
            width,
            height,
            frame_area,
            center_trail,
            frame_idx,
            max_trail_frames,
        )

    print("Start tracking...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        ref_box = last_confirmed_box if occluded or lost_count > 0 else target_box
        pred_box = predict_next_box(ref_box, velocity, width, height)

        detections = detect_persons(model, frame)

        if len(detections) > 0:
            if occluded or lost_count > 0:
                nearby = detections
            else:
                nearby = choose_candidates_near_prediction(detections, pred_box, width, height)

            scored = []

            for det in nearby:
                s = score_candidate(
                    det["box"],
                    pred_box,
                    target_profile,
                    frame,
                    frame_area,
                    occluded=occluded
                )

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

            if not occluded:
                is_ambiguous = detect_ambiguity(scored, width, height)

                if len(scored) > 0:
                    ok_track, track_app_score = passes_tracking_gate(scored[0])
                else:
                    ok_track, track_app_score = False, 0.0

                if (
                    len(scored) > 0
                    and (not is_ambiguous)
                    and scored[0]["score"] >= NORMAL_ACCEPT_THRES
                    and ok_track
                ):
                    best = scored[0]
                    new_box = best["box"]

                    old_center = np.array(box_center(last_confirmed_box), dtype=np.float32)
                    new_center = np.array(box_center(new_box), dtype=np.float32)
                    velocity = 0.70 * velocity + 0.30 * (new_center - old_center)

                    target_box = new_box
                    last_confirmed_box = new_box.copy()

                    target_profile = blend_profile(
                        target_profile,
                        best["candidate_profile"],
                        PROFILE_UPDATE_ALPHA
                    )

                    lost_count = 0
                    occluded = False
                    reacquire_box = None
                    reacquire_count = 0

                    tracking_state = "TRACKING"
                    display_score = track_app_score
                    detail_text = (
                        f"U:{best['upper']:.2f} "
                        f"L:{best['lower']:.2f} "
                        f"H:{best['head']:.2f} "
                        f"G:{best['global']:.2f}"
                    )

                    center_trail = add_center_to_trail(
                        center_trail,
                        last_confirmed_box,
                        frame_idx,
                        max_trail_frames
                    )

                else:
                    occluded = True
                    lost_count += 1
                    velocity[:] = 0.0

                    target_box = last_confirmed_box.copy()
                    tracking_state = "LOST"
                    display_score = 0.0
                    detail_text = ""

                    reacquire_box = None
                    reacquire_count = 0

                    center_trail = prune_center_trail(
                        center_trail,
                        frame_idx,
                        max_trail_frames
                    )

            else:
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

                if len(passed_candidates) >= 2:
                    margin_ok = (
                        passed_candidates[0]["app_score"] - passed_candidates[1]["app_score"]
                    ) >= REID_MARGIN_THRES
                else:
                    margin_ok = len(passed_candidates) == 1

                if len(passed_candidates) > 0 and margin_ok:
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
                        old_center = np.array(box_center(last_confirmed_box), dtype=np.float32)
                        new_center = np.array(box_center(candidate_box), dtype=np.float32)
                        velocity = 0.50 * velocity + 0.50 * (new_center - old_center)

                        target_box = candidate_box
                        last_confirmed_box = candidate_box.copy()

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

                        center_trail = add_center_to_trail(
                            center_trail,
                            last_confirmed_box,
                            frame_idx,
                            max_trail_frames
                        )

                    else:
                        lost_count += 1
                        velocity[:] = 0.0
                        target_box = last_confirmed_box.copy()

                        tracking_state = "LOST"
                        display_score = 0.0
                        detail_text = ""

                        center_trail = prune_center_trail(
                            center_trail,
                            frame_idx,
                            max_trail_frames
                        )

                else:
                    lost_count += 1
                    velocity[:] = 0.0
                    target_box = last_confirmed_box.copy()

                    tracking_state = "LOST"
                    display_score = 0.0
                    detail_text = ""

                    center_trail = prune_center_trail(
                        center_trail,
                        frame_idx,
                        max_trail_frames
                    )

        else:
            occluded = True
            lost_count += 1
            velocity[:] = 0.0
            target_box = last_confirmed_box.copy()

            tracking_state = "LOST"
            display_score = 0.0
            detail_text = ""

            center_trail = prune_center_trail(
                center_trail,
                frame_idx,
                max_trail_frames
            )

        lost_count = min(lost_count, MAX_LOST_FRAMES)

        # -------------------------------------------------
        # Draw result / 绘制结果
        # -------------------------------------------------
        if tracking_state == "LOST":
            vis = frame.copy()
            vis = draw_lost_message(vis, lost_count)
            vis = draw_center_trail(vis, center_trail)

        else:
            if tracking_state == "REID_BY_FEATURE":
                label = f"ID:1 REID_BY_FEATURE A:{display_score:.2f}"
            else:
                label = f"ID:1 TRACKING A:{display_score:.2f}"

            vis = frame.copy()
            vis = draw_target(vis, last_confirmed_box, label=label, detail=detail_text)
            vis = draw_center_trail(vis, center_trail)

        writer.write(vis)

    cap.release()
    writer.release()
    print(f"Done. Result saved to: {output_path}")


if __name__ == "__main__":
    main()