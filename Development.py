from flask import Flask, Response, jsonify
from flask_sock import Sock
from ultralytics import YOLO
import cv2
import threading
import subprocess
import serial
import serial.tools.list_ports
import random
import time
import glob
import os
import numpy as np

app  = Flask(__name__)
sock = Sock(app)

# ── Detection model ───────────────────────────────────────────────────────────
model = YOLO("yolov8n.pt")

# ── Webcam Auto-Detection ─────────────────────────────────────────────────────
def find_working_cameras():
    working = []
    for i in range(10):
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            for _ in range(5):
                ok, _ = cam.read()
                if ok:
                    print(f"[Camera] Found working camera at index {i}")
                    working.append(i)
                    break
                time.sleep(0.1)
            cam.release()
    return working

working_cams = find_working_cameras()
if len(working_cams) >= 2:
    front_cam_idx = working_cams[0]
    ceiling_cam_idx = working_cams[1]
elif len(working_cams) == 1:
    front_cam_idx = working_cams[0]
    ceiling_cam_idx = working_cams[0]
else:
    front_cam_idx = 0
    ceiling_cam_idx = 1
    print("[Camera] No working cameras found in indices 0-9. Check USB.")

def init_cam(idx, w, h, fps):
    c = cv2.VideoCapture(idx)
    c.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    c.set(cv2.CAP_PROP_FPS, fps)
    return c

# ── Audio ─────────────────────────────────────────────────────────────────────
AUDIO_FOLDER      = os.path.expanduser("~/audioclips")
MANUAL_FOLDER     = os.path.expanduser("~/audioclips-manual")
VOICELINES_FOLDER = os.path.expanduser("~/voicelines")
PROXIMITY_THRESHOLD = 0.45
AUDIO_COOLDOWN      = 20

# Which folder is used as the random-cue pool: "auto" (audioclips) or "voicelines"
active_random_source      = "auto"
active_random_source_lock = threading.Lock()

# Shuffle-bag: play every clip once before repeating
_shuffle_bag      = []   # remaining clips in current shuffle cycle
_shuffle_bag_lock = threading.Lock()

detectr_enabled = False
detectr_lock    = threading.Lock()
last_audio_time = 0
audio_lock      = threading.Lock()

# Set True while a person is close enough to block the robot's path
person_blocking      = False
person_blocking_lock = threading.Lock()
# Fraction of frame height a bounding box must reach to count as "blocking"
BLOCKING_THRESHOLD = 0.85  # Person must fill 85% of frame height to block path

# ── Locks ─────────────────────────────────────────────────────────────────────
camera_index_lock = threading.Lock()
log_file_lock     = threading.Lock()

# ── Detection Tracking ────────────────────────────────────────────────────────
active_tracks   = {}
next_track_id   = 0
tracking_lock   = threading.Lock()
TRACK_MAX_DIST  = 100  # Max pixel distance to match centroids between cycles
TRACK_MAX_LOST  = 2    # How many detection cycles to wait before dropping a track

# ── Camera frame state ────────────────────────────────────────────────────────
current_frame = None
frame_lock    = threading.Lock()
front_cam_enabled   = True
ceiling_cam_enabled = True
cam_enabled_lock    = threading.Lock()

# ── AutoNav Logic ─────────────────────────────────────────────────────────────
class NavState:
    running = False
    lock = threading.Lock()

nav_state = NavState()

# ── Person Detection Log ───────────────────────────────────────────────────────
import json
LOG_FILE = os.path.expanduser("~/person_detections.json")
current_hallway = "Unknown"
current_light_counter = 0
nav_location_lock = threading.Lock()

def log_person_detection(count=1):
    """Append a person detection event to the persistent JSON log."""
    with nav_location_lock:
        hallway = current_hallway
        light = current_light_counter
    entry = {
        "Hallway": hallway,
        "light_number": light,
        "date_and_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "count": count
    }
    try:
        with log_file_lock:
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r") as f:
                    content = f.read().strip()
                    data = json.loads(content) if content else []
            else:
                data = []
            data.append(entry)
            with open(LOG_FILE, "w") as f:
                json.dump(data, f)
    except Exception as e:
        print(f"[Log] Failed to write detection log: {e}")

# ── Ceiling Camera (Localization) ─────────────────────────────────────────────
ceiling_frame      = None
ceiling_frame_lock = threading.Lock()

# Topological map: corner_id → navigation action
TOPOLOGICAL_MAP = {
    "600500": {"action": "TURN_LEFT"},
    "500200": {"action": "TURN_LEFT"},
    "200300": {"action": "TURN_LEFT"},
    "300600": {"action": "TURN_LEFT"}
}
# 300400 (brown circle) and 500400 (yellow square) are intentionally omitted 
# from TOPOLOGICAL_MAP based on "turn left when it detects any circle except brown".

MARKERS = [
    {"id": "600500", "color": "pink",   "shape": "circle",    "ranges": [((0, 40, 60), (15, 200, 255)), ((150, 40, 60), (180, 200, 255))]},
    {"id": "500400", "color": "yellow", "shape": "rectangle", "ranges": [((20, 120, 80), (35, 255, 255))]},
    {"id": "500200", "color": "yellow", "shape": "circle",    "ranges": [((20, 120, 80), (35, 255, 255))]},
    {"id": "200300", "color": "blue",   "shape": "circle",    "ranges": [((100, 120, 80), (130, 255, 255))]},
    {"id": "300400", "color": "brown",  "shape": "circle",    "ranges": [((5, 60, 20), (25, 200, 120))]},
    {"id": "300600", "color": "green",  "shape": "circle",    "ranges": [((35, 50, 50), (90, 255, 255))]}
]

MARKER_MIN_AREA = 400       # Increased from 150 to prevent tiny noise from triggering turns
CIRCULARITY_THRESH = 0.75  # Increased from 0.55 to demand actual round shapes, ignoring stretched lens flares

def color_mask(hsv, ranges):
    mask = None
    for (lo, hi) in ranges:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask

def classify_contour(contour):
    area = cv2.contourArea(contour)
    if area < MARKER_MIN_AREA:
        return None
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    if circularity >= CIRCULARITY_THRESH:
        return "circle"
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    if 4 <= len(approx) <= 6:
        return "rectangle"
    return None

def detect_ceiling_markers(frame_bgr):
    """
    Returns:
        corner_id (str|None) — id of a marker seen overhead
        light_seen (bool) — whether a ceiling light panel was seen
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (7, 7), 0)
    
    corner_id = None
    for marker in MARKERS:
        mask = color_mask(hsv, marker["ranges"])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            shape = classify_contour(cnt)
            if shape == marker["shape"]:
                corner_id = marker["id"]
                break
        if corner_id:
            break
    return corner_id


def compute_centering_turn(frame_gray):
    """
    Tracks the floor path using the brightest region that touches the bottom
    of the frame. Focused on the immediate floor (bottom 35%) and masked on the 
    sides to ignore window glare and distant hallway branches.
    """
    h, w = frame_gray.shape
    roi_start_y = int(h * 0.65)
    roi = frame_gray[roi_start_y:h, :].copy()
    roi_h = h - roi_start_y

    # Mask horizontal sides significantly to ignore window reflections and 
    # intersecting hallway branches before reaching the corner
    side_margin = int(w * 0.25)
    roi[:, :side_margin] = 0
    roi[:, w-side_margin:] = 0

    blurred = cv2.GaussianBlur(roi, (25, 25), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological ops to clean glare and close gaps
    kernel = np.ones((9, 9), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for c in contours:
        if cv2.contourArea(c) > 800:  # Ignore distant or small patches
            _, cy, _, ch = cv2.boundingRect(c)
            # Must touch the bottom of our focused ROI
            if cy + ch >= roi_h - 5:
                valid_contours.append(c)

    if valid_contours:
        target = max(valid_contours, key=cv2.contourArea)
        M = cv2.moments(target)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            # Small rightward bias (+3%) to correct persistent left drift
            error = (cX - (w / 2)) / (w / 2) - 0.03
            
            # Proportional steering
            turn = int(error * 25)
            return max(-30, min(30, turn))
    return 0

def execute_turn(action):
    """Executes a calibrated turn with serial error checking."""
    if action == "TURN_LEFT":
        cmd = "DRIVE 0 -70"
    elif action == "TURN_RIGHT":
        cmd = "DRIVE 0 70"
    else:
        return

    print(f"[Nav] Executing: {action}")
    res = robot_send(cmd)
    
    if res.startswith("ERR"):
        print(f"[Nav] Turn command failed ({res}). Safety stop.")
        robot_send("STOP")
        return

    time.sleep(0.55)
    robot_send("STOP")

def navigation_loop():
    global current_hallway, current_light_counter
    cooldown_frames   = 0
    armed_landmark    = None
    armed_frame_count = 0          # counts frames since landmark was armed
    APPROACH_FRAMES   = 0          # Instant: trigger turn as soon as symbol is confirmed
    stuck_frames      = 0
    prev_gray         = None
    last_block_print  = 0
    # Consecutive-frame confirmation (prevents single-frame noise from triggering)
    last_corner_seen    = None
    consecutive_corners = 0
    CORNER_CONFIRM_FRAMES = 3  # Must see the same marker this many frames in a row

    while True:
        with nav_state.lock: is_running = nav_state.running
        with frame_lock:    frame      = current_frame

        if not is_running or frame is None:
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray  = clahe.apply(gray)

        try:
            turn = compute_centering_turn(gray)
        except:
            turn = 0
        is_intersection = False

        # ── Ceiling marker detection ───────────────────────────────────────────
        with ceiling_frame_lock: c_frame = ceiling_frame
        corner_seen = None
        if c_frame is not None:
            try:
                corner_seen = detect_ceiling_markers(c_frame)
            except Exception as e:
                print(f"[Ceiling] Detection error: {e}")

        # ── Consecutive-frame confirmation ─────────────────────────────────────
        if corner_seen == last_corner_seen:
            consecutive_corners += 1
        else:
            last_corner_seen    = corner_seen
            consecutive_corners = 1
        confirmed_corner = corner_seen if consecutive_corners >= CORNER_CONFIRM_FRAMES else None

        # ── Hallway logic ──────────────────────────────────────────────────────
        # Only update hallway when NOT in turn cooldown to prevent flickering
        if cooldown_frames == 0 and confirmed_corner:
            new_hallway = confirmed_corner[3:]
            with nav_location_lock:
                if current_hallway != new_hallway:
                    current_hallway = new_hallway
                    current_light_counter = 0
                    print(f"[Nav] Entered hallway {current_hallway}, light counter reset.")

        # ── Corner / turn logic ────────────────────────────────────────────────
        if cooldown_frames > 0:
            cooldown_frames -= 1
            # Reset confirmation counter during cooldown so the robot must
            # see the marker fresh after a turn — prevents stale frames from
            # re-arming immediately when cooldown expires.
            last_corner_seen    = None
            consecutive_corners = 0
        else:
            if armed_landmark:
                armed_frame_count += 1
                print(f"[NavCeiling] Approaching corner {armed_landmark} — frame {armed_frame_count}/{APPROACH_FRAMES}")
                if armed_frame_count >= APPROACH_FRAMES:
                    # Enough time has passed to reach the corner — execute the turn.
                    # Ignore person_blocking here: the robot is committed to the corner
                    # and has already stopped; a person in the threshold zone at this
                    # instant should not abort the turn.
                    print(f"[NavCeiling] Executing turn for: {armed_landmark}")
                    robot_send("STOP")
                    time.sleep(0.55)
                    if armed_landmark in TOPOLOGICAL_MAP:
                        execute_turn(TOPOLOGICAL_MAP[armed_landmark]["action"])
                    time.sleep(0.5)
                    cooldown_frames   = 90
                    armed_landmark    = None
                    armed_frame_count = 0
            elif confirmed_corner:
                if confirmed_corner in TOPOLOGICAL_MAP:
                    armed_landmark    = confirmed_corner
                    armed_frame_count = 0
                    print(f"[NavCeiling] Armed turn for: {armed_landmark}")

        try:
            with person_blocking_lock: blocked = person_blocking
            if blocked:
                robot_send("STOP")
                now = time.time()
                if now - last_block_print > 3:
                    print("[Nav] Person blocking path — waiting...")
                    last_block_print = now
                stuck_frames = 0
            else:
                if is_running and abs(turn) < 15:
                    if prev_gray is not None:
                        diff = cv2.absdiff(gray, prev_gray)
                        if np.mean(diff) < 1.5:
                            stuck_frames += 1
                        else:
                            stuck_frames = 0
                        
                        if stuck_frames > 40:
                            print("[Nav] Stuck detected! Executing shake-off maneuver...")
                            robot_send("STOP")
                            time.sleep(0.2)
                            robot_send("DRIVE 0 -70") # Shake left
                            time.sleep(0.35)
                            robot_send("DRIVE 0 70")  # Shake right
                            time.sleep(0.35)
                            robot_send("STOP")
                            time.sleep(0.2)
                            stuck_frames = 0
                else:
                    stuck_frames = 0
                prev_gray = gray.copy()

                res = robot_send(f"DRIVE 70 {turn}")
                if res.startswith("ERR"): robot_connect()
        except: pass
        time.sleep(0.1)

# ── Robot serial ──────────────────────────────────────────────────────────────
robot_ser  = None
robot_lock = threading.Lock()
ROBOT_PORT = "/dev/ttyACM0"
ROBOT_BAUD = 115200

# ── External mic card ─────────────────────────────────────────────────────────
EXT_MIC_CARD = 3

# ─── Robot helpers ────────────────────────────────────────────────────────────
def find_vex_port():
    VEX_VID = 0x2888
    for p in serial.tools.list_ports.comports():
        # Priority 1: Check Vendor ID for VEX
        if p.vid == VEX_VID: return p.device
        
        # Priority 2: Linux-style serial devices
        if p.device.startswith("/dev/ttyACM") or p.device.startswith("/dev/ttyUSB"): return p.device
        
        # Priority 3: Windows-style COM ports
        if p.device.startswith("COM"): return p.device
    return None

def robot_connect():
    global robot_ser
    try:
        with robot_lock:
            if robot_ser and robot_ser.is_open: return True, "Already connected"
            port = find_vex_port() or ROBOT_PORT
            robot_ser = serial.Serial(port, ROBOT_BAUD, timeout=0.1)
            time.sleep(1)
        return True, f"Connected on {port}"
    except Exception as e:
        robot_ser = None
        return False, str(e)

def robot_send(cmd: str) -> str:
    with robot_lock:
        if not robot_ser or not robot_ser.is_open: return "ERR not connected"
        try:
            robot_ser.write((cmd + "\n").encode())
            return "OK"
        except Exception as e: return f"ERR {e}"

# ─── Audio helpers ────────────────────────────────────────────────────────────
def get_clips_from(folder):
    return sorted(
        glob.glob(os.path.join(folder, "*.mp3")) +
        glob.glob(os.path.join(folder, "*.wav")) +
        glob.glob(os.path.join(folder, "*.m4a"))
    )

def stop_audio():
    subprocess.run(["pkill", "-x", "mpg123"],  stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-x", "aplay"],   stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-x", "ffplay"],  stderr=subprocess.DEVNULL)

def play_clip(path):
    stop_audio()
    ext = os.path.splitext(path)[1].lower()
    print(f"Playing: {os.path.basename(path)}")
    if ext == ".mp3":  subprocess.Popen(["mpg123", "-q", path])
    elif ext == ".wav": subprocess.Popen(["aplay",  "-q", path])
    elif ext == ".m4a": subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path])

def play_random_clip():
    global _shuffle_bag
    with active_random_source_lock:
        src = active_random_source
    folder = VOICELINES_FOLDER if src == "voicelines" else AUDIO_FOLDER
    clips = get_clips_from(folder)
    if not clips:
        return
    with _shuffle_bag_lock:
        # Refill and reshuffle when the bag is empty or the folder changed
        remaining = [c for c in _shuffle_bag if c in clips]
        if not remaining:
            remaining = clips[:]
            random.shuffle(remaining)
        next_clip = remaining.pop(0)
        _shuffle_bag = remaining
    play_clip(next_clip)

def try_trigger_audio(box_h, frame_h):
    global last_audio_time
    with detectr_lock:
        if not detectr_enabled: return
    if (box_h / frame_h) < PROXIMITY_THRESHOLD: return
    with audio_lock:
        now = time.time()
        if now - last_audio_time < AUDIO_COOLDOWN: return
        last_audio_time = now
    threading.Thread(target=play_random_clip, daemon=True).start()

# ─── Mic session ─────────────────────────────────────────────────────────────
def run_mic_session(ws):
    print("[Mic] Starting ffmpeg session")
    proc = subprocess.Popen(
        ["ffmpeg", "-f", "webm", "-i", "pipe:0", "-f", "alsa", "plughw:0,0", "-ar", "44100"],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    try:
        while True:
            data = ws.receive()
            if data is None: break
            if isinstance(data, (bytes, bytearray)):
                try:
                    proc.stdin.write(bytes(data))
                    proc.stdin.flush()
                except Exception: break
    except Exception: pass
    finally:
        try: proc.stdin.close(); proc.wait(timeout=2)
        except Exception: proc.kill()
        print("[Mic] ffmpeg session closed")

# ─── Camera threads ───────────────────────────────────────────────────────────
def capture_frames():
    global current_frame, front_cam_idx
    my_idx = front_cam_idx
    camera = init_cam(my_idx, 480, 360, 30)
    fail_count = 0
    while True:
        with camera_index_lock:
            target_idx = front_cam_idx
            
        if my_idx != target_idx:
            try: camera.release()
            except: pass
            my_idx = target_idx
            camera = init_cam(my_idx, 480, 360, 30)
            fail_count = 0
            
        ok, frame = camera.read()
        if ok:
            fail_count = 0
            with frame_lock: current_frame = frame
        else:
            fail_count += 1
            if fail_count >= 10:
                print(f"[Camera] Read failed {fail_count} times, reinitializing...")
                try: camera.release()
                except Exception: pass
                time.sleep(2)
                camera = init_cam(my_idx, 480, 360, 30)
                fail_count = 0
            else: time.sleep(0.1)

def capture_ceiling_frames():
    """Continuously grab frames from the upward-facing ceiling camera.
    
    Handles USB disconnects robustly:
    - Re-scans for available cameras on each reinit attempt so that if
      Linux reassigns the /dev/video* node after a reconnect, it is found.
    - Uses exponential backoff (2s → 4s → 8s → max 30s) to avoid CPU/RAM
      thrashing that can trigger the OOM killer.
    """
    global ceiling_frame, ceiling_cam_idx
    my_idx = ceiling_cam_idx
    ceiling_cam = init_cam(my_idx, 320, 240, 15)
    fail_count   = 0
    retry_delay  = 2  # seconds; doubles on each failed reinit, capped at 30

    while True:
        # Honour a manual camera swap from the web UI
        with camera_index_lock:
            target_idx = ceiling_cam_idx
            
        if my_idx != target_idx:
            try: ceiling_cam.release()
            except: pass
            my_idx = target_idx
            ceiling_cam = init_cam(my_idx, 320, 240, 15)
            fail_count  = 0
            retry_delay = 2

        ok, frame = ceiling_cam.read()
        if ok:
            fail_count  = 0
            retry_delay = 2  # reset backoff on success
            with ceiling_frame_lock: ceiling_frame = frame
        else:
            fail_count += 1
            if fail_count >= 10:
                print(f"[CeilingCam] Lost connection — waiting {retry_delay}s before reinit...")
                try: ceiling_cam.release()
                except Exception: pass

                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)  # exponential backoff, cap at 30s

                # Re-scan: the camera may have reconnected on a different index
                print("[CeilingCam] Scanning for cameras...")
                cams = find_working_cameras()
                with camera_index_lock:
                    if len(cams) >= 2:
                        ceiling_cam_idx = cams[1]
                    elif len(cams) == 1:
                        ceiling_cam_idx = cams[0]
                    my_idx = ceiling_cam_idx
                
                print(f"[CeilingCam] Reinitializing on index {my_idx}...")
                ceiling_cam = init_cam(my_idx, 320, 240, 15)
                fail_count  = 0
            else:
                time.sleep(0.1)

def generate_raw_feed():
    BLACK = cv2.imencode('.jpg', np.zeros((360, 480, 3), np.uint8))[1].tobytes()
    while True:
        with cam_enabled_lock: enabled = front_cam_enabled
        if not enabled:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + BLACK + b'\r\n'
            time.sleep(0.1)
            continue
        with frame_lock:
            if current_frame is None: continue
            frame = current_frame.copy()
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok: yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'

def generate_detection_feed():
    global person_blocking
    BLACK = cv2.imencode('.jpg', np.zeros((360, 480, 3), np.uint8))[1].tobytes()
    count, annotated = 0, None
    last_logged_time = 0
    while True:
        with cam_enabled_lock: enabled = front_cam_enabled
        if not enabled:
            with person_blocking_lock: person_blocking = False
            with tracking_lock: active_tracks.clear()
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + BLACK + b'\r\n'
            time.sleep(0.1)
            continue
        with frame_lock:
            if current_frame is None: continue
            frame = current_frame.copy()
        if count % 15 == 0:
            fh, fw = frame.shape[:2]
            results = model(frame, classes=[0], verbose=False, imgsz=320)
            annotated = results[0].plot()
            
            close_person = False
            current_cycle_tracks = [] # (track_id, box, centroid)
            
            with tracking_lock:
                # 1. Process detections and match to existing tracks
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    box_h = float(y2 - y1)
                    distance_pct = box_h / fh
                    
                    try_trigger_audio(box_h, fh)
                    
                    # Safety Stop: Only block if the person is close AND in the central path
                    # (central 70% of frame width)
                    if distance_pct >= BLOCKING_THRESHOLD:
                        if x1 < fw * 0.85 and x2 > fw * 0.15:
                            close_person = True
                        
                    # Find best match in existing tracks
                    best_id = None
                    min_dist = TRACK_MAX_DIST
                    for tid, tdata in active_tracks.items():
                        if tid in [t[0] for t in current_cycle_tracks]: continue
                        dist = np.sqrt((cx - tdata['centroid'][0])**2 + (cy - tdata['centroid'][1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_id = tid
                    
                    if best_id is not None:
                        active_tracks[best_id].update({
                            'centroid': (cx, cy),
                            'last_box': (x1, y1, x2, y2),
                            'last_seen': count,
                            'lost_count': 0
                        })
                        current_cycle_tracks.append((best_id, (x1, y1, x2, y2), (cx, cy)))
                    else:
                        # New track
                        global next_track_id
                        new_id = next_track_id
                        next_track_id += 1
                        active_tracks[new_id] = {
                            'centroid': (cx, cy),
                            'last_box': (x1, y1, x2, y2),
                            'last_seen': count,
                            'lost_count': 0
                        }
                        current_cycle_tracks.append((new_id, (x1, y1, x2, y2), (cx, cy)))
                        # Log new person immediately
                        threading.Thread(target=log_person_detection, args=(1,), daemon=True).start()

                # 2. Cleanup lost tracks and handle edge-exit
                to_delete = []
                for tid, tdata in active_tracks.items():
                    if tid not in [t[0] for t in current_cycle_tracks]:
                        tdata['lost_count'] += 1
                        
                        # Check if they were near the edge when last seen
                        bx1, by1, bx2, by2 = tdata['last_box']
                        near_edge = (bx1 < 30 or bx2 > fw - 30 or by1 < 30 or by2 > fh - 30)
                        
                        if near_edge or tdata['lost_count'] >= TRACK_MAX_LOST:
                            to_delete.append(tid)
                
                for tid in to_delete:
                    del active_tracks[tid]
            
            with person_blocking_lock:
                person_blocking = close_person
        elif annotated is None: annotated = frame
        count += 1
        ok, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok: yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/api/nav_status')
def api_nav_status():
    with nav_location_lock:
        hallway = current_hallway
        lights = current_light_counter
    return jsonify({
        "hallway": hallway,
        "lights": lights
    })

@app.route('/api/temp')
def api_temp():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp_c = int(f.read().strip()) / 1000.0
        return jsonify({"temp": f"{temp_c:.1f}°C"})
    except:
        return jsonify({"temp": "Unknown"})

@app.route('/camera/<cam>/<state>')
def toggle_camera(cam, state):
    global front_cam_enabled, ceiling_cam_enabled
    enabled = (state == 'on')
    with cam_enabled_lock:
        if cam == 'front':
            front_cam_enabled = enabled
        elif cam == 'ceiling':
            ceiling_cam_enabled = enabled
        else:
            return jsonify({"message": f"Unknown camera: {cam}"}), 400
    return jsonify({"camera": cam, "enabled": enabled,
                    "message": f"{cam.capitalize()} camera {'enabled' if enabled else 'disabled'}"})

@app.route('/raw')
def raw(): return Response(generate_raw_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection')
def detection(): return Response(generate_detection_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/swap_cameras')
def swap_cameras():
    global front_cam_idx, ceiling_cam_idx
    with camera_index_lock:
        front_cam_idx, ceiling_cam_idx = ceiling_cam_idx, front_cam_idx
    return jsonify({"message": "Cameras swapped successfully!"})

@app.route('/detectr/<state>')
def set_detectr(state):
    global detectr_enabled
    with detectr_lock: detectr_enabled = (state == 'on')
    return jsonify({"message": f"Automatic Audio Cues {'on' if detectr_enabled else 'off'}.", "enabled": detectr_enabled})

@app.route('/api/detections')
def api_detections():
    try:
        with log_file_lock:
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r") as f:
                    content = f.read().strip()
                    data = json.loads(content) if content else []
            else:
                data = []
        return jsonify({"detections": data})
    except Exception:
        return jsonify({"detections": []})

@app.route('/play')
def play_random():
    with active_random_source_lock:
        src = active_random_source
    folder = VOICELINES_FOLDER if src == "voicelines" else AUDIO_FOLDER
    clips = get_clips_from(folder)
    label = "voicelines" if src == "voicelines" else "audioclips"
    if not clips: return jsonify({"message": f"No clips in ~/{label}"})
    threading.Thread(target=play_random_clip, daemon=True).start()
    return jsonify({"message": f"Playing random clip from {label}..."})

@app.route('/play/<folder>/<filename>')
def play_specific(folder, filename):
    if folder == "auto":        base = AUDIO_FOLDER
    elif folder == "voicelines": base = VOICELINES_FOLDER
    else:                       base = MANUAL_FOLDER
    path = os.path.join(base, filename)
    if not os.path.exists(path): return jsonify({"message": f"Not found: {filename}"})
    threading.Thread(target=play_clip, args=(path,), daemon=True).start()
    return jsonify({"message": f"Playing: {filename}"})

@app.route('/stop')
def stop():
    stop_audio()
    return jsonify({"message": "Audio stopped."})

@app.route('/api/set_random_source/<source>')
def set_random_source(source):
    global active_random_source
    if source not in ("auto", "voicelines"):
        return jsonify({"message": f"Unknown source '{source}'"}), 400
    with active_random_source_lock:
        active_random_source = source
    label = "Voice Lines" if source == "voicelines" else "Audio Clips"
    return jsonify({"source": source, "message": f"Random cue source: {label}"})

@app.route('/clips')
def clips():
    auto       = [{"name": os.path.basename(p), "folder": "auto"}       for p in get_clips_from(AUDIO_FOLDER)]
    manual     = [{"name": os.path.basename(p), "folder": "manual"}     for p in get_clips_from(MANUAL_FOLDER)]
    voicelines = [{"name": os.path.basename(p), "folder": "voicelines"} for p in get_clips_from(VOICELINES_FOLDER)]
    return jsonify({"clips": auto + manual + voicelines})

@app.route('/volume/<int:level>')
def set_volume(level):
    level = max(0, min(100, level))
    subprocess.run(["amixer", "sset", "PCM", "-M", f"{level}%"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    return jsonify({"message": f"Volume set to {level}%"})

@app.route('/robot/connect')
def r_connect():
    ok, msg = robot_connect()
    return jsonify({"ok": ok, "message": msg})

@app.route('/robot/stop')
def r_stop():
    return jsonify({"message": robot_send("STOP")})

@app.route('/api/start')
def api_start():
    with nav_state.lock: nav_state.running = True
    ok, msg = robot_connect()
    if not ok: return jsonify({"message": f"AutoNav Started, BUT {msg}"})
    return jsonify({"message": f"AutoNav Started - {msg}"})

@app.route('/api/stop')
def api_stop():
    with nav_state.lock: nav_state.running = False
    robot_send("STOP")
    return jsonify({"message": "AutoNav Paused"})

@sock.route('/mic')
def mic_ws(ws):
    print("[Mic] Client connected")
    run_mic_session(ws)
    print("[Mic] Client disconnected")

@sock.route('/ext_audio')
def ext_audio_ws(ws):
    print("[ExtAudio] Client connected")
    proc = subprocess.Popen(
        ["ffmpeg", "-f", "alsa", "-i", f"plughw:{EXT_MIC_CARD},0", "-f", "s16le", "-ar", "22050", "-ac", "1", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    try:
        while True:
            chunk = proc.stdout.read(2048)
            if not chunk: break
            ws.send(chunk)
    except Exception: pass
    finally:
        try: proc.kill(); proc.stdout.close(); proc.wait(timeout=2)
        except Exception: pass
        print("[ExtAudio] Client disconnected")

# ─── Main page ────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return r'''<!DOCTYPE html>
<html>
<head>
<title>Person Detection & AutoNav</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#fff;font-family:sans-serif;
     min-height:100vh;padding:24px;
     display:flex;flex-direction:column;align-items:center;gap:24px}
h1{font-size:1.4rem;color:#eee;letter-spacing:1px}

/* top row — feeds + clip panel side by side */
.top-row{display:flex;gap:20px;justify-content:center;align-items:flex-start;flex-wrap:wrap;width:100%}
.feeds{display:flex;flex-direction:column;gap:16px;align-items:center}
.feeds-inner{display:flex;flex-direction:column;gap:16px;align-items:center}
.feed-box{display:flex;flex-direction:column;align-items:center;gap:10px;width:100%}
.feed-box h2{font-size:.85rem;text-transform:uppercase;letter-spacing:2px}
.feed-box img{border:2px solid #333;border-radius:8px;width:100%;max-width:640px}
.live-lbl{color:#4caf50}.det-lbl{color:#f44336}

/* toggle */
.toggle-row{display:flex;align-items:center;gap:14px;
  background:#1a1a1a;border:1px solid #333;border-radius:10px;padding:14px 24px}
.toggle-name{font-size:1rem;font-weight:700;letter-spacing:1px;
  text-transform:uppercase;color:#aaa;transition:color .3s}
.toggle-name.on{color:#f44336}
.toggle-sub{font-size:.78rem;color:#555;transition:color .3s}
.toggle-sub.on{color:#f44336}
.switch{position:relative;display:inline-block;width:56px;height:28px}
.switch input{opacity:0;width:0;height:0}
.slider{position:absolute;cursor:pointer;inset:0;
  background:#333;border-radius:28px;transition:background .3s}
.slider:before{position:absolute;content:"";height:20px;width:20px;
  left:4px;bottom:4px;background:#fff;border-radius:50%;transition:transform .3s}
input:checked+.slider{background:#f44336}
input:checked+.slider:before{transform:translateX(28px)}

/* buttons */
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;justify-content:center}
button{border:none;padding:11px 24px;font-size:.95rem;border-radius:6px;
  cursor:pointer;font-weight:600;transition:background .15s,transform .1s}
button:active{transform:scale(.96)}
.btn-blue{background:#1e88e5;color:#fff}.btn-blue:hover{background:#1565c0}
.btn-red {background:#e53935;color:#fff}.btn-red:hover {background:#b71c1c}
.btn-grey{background:#333;color:#fff;border:2px solid #555}
.btn-grey.on{background:#e53935;border-color:#e53935}
.btn-green{background:#43a047;color:#fff}.btn-green:hover{background:#2e7d32}

#audioStatus{font-size:.82rem;color:#aaa;min-height:18px;text-align:center}

/* clip list — sidebar next to feeds */
.clip-panel{width:360px;min-width:280px;display:flex;flex-direction:column;gap:8px}
.clip-panel h3{font-size:.85rem;text-transform:uppercase;letter-spacing:2px;
  color:#aaa;text-align:center}
/* Accordion */
.accordion { background-color: #333; color: #fff; cursor: pointer; padding: 14px; width: 100%; border: none; text-align: left; outline: none; font-size: .95rem; transition: 0.4s; border-radius: 4px; margin-bottom: 2px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;}
.accordion.active, .accordion:hover { background-color: #1e88e5; }
.panel { padding: 0 6px; background-color: #1a1a1a; max-height: 0; overflow-y: auto; transition: max-height 0.2s ease-out; display: flex; flex-direction: column; gap: 4px; margin-bottom: 10px; }
.panel::-webkit-scrollbar{width:6px}
.panel::-webkit-scrollbar-track{background:#222;border-radius:4px}
.panel::-webkit-scrollbar-thumb{background:#444;border-radius:4px}
.clip-btn{background:#252525;border:1px solid #333;border-radius:5px;
  padding:12px 16px;cursor:pointer;font-size:.95rem;color:#ddd;
  text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  width:100%;transition:background .15s;flex-shrink:0}
.clip-btn:hover{background:#1e88e5;color:#fff;border-color:#1e88e5}
.clip-btn.playing{background:#1565c0;color:#fff;border-color:#1565c0}
.sec-lbl{font-size:.75rem;text-transform:uppercase;letter-spacing:2px;
  color:#555;padding:8px 8px 2px}
.badge-mp3{color:#f9a825;font-size:.75rem;margin-right:5px}
.badge-wav{color:#66bb6a;font-size:.75rem;margin-right:5px}
.badge-man{color:#ab47bc;font-size:.75rem;margin-right:5px}
.no-clips{color:#555;font-size:.85rem;text-align:center;padding:16px}
/* volume */
.vol-row{display:flex;align-items:center;gap:12px;justify-content:center}
.vol-row label{font-size:.82rem;color:#aaa;min-width:56px}
input.vol-slider{width:180px;accent-color:#1e88e5}
#volVal{font-size:.82rem;color:#eee;min-width:36px}

/* ── Robot panel ── */
.robot-panel{width:100%;max-width:560px;background:#1a1a1a;
  border:1px solid #333;border-radius:10px;padding:20px;
  display:flex;flex-direction:column;align-items:center;gap:16px}
.robot-panel h3{font-size:.85rem;text-transform:uppercase;
  letter-spacing:2px;color:#aaa}
.conn-row{display:flex;align-items:center;gap:10px}
.dot{width:10px;height:10px;border-radius:50%;background:#555;transition:background .3s}
.dot.conn{background:#4caf50}.dot.pi{background:#1e88e5}
#connText{font-size:.82rem;color:#aaa}

#robotStatus{font-size:.82rem;color:#aaa;min-height:18px;text-align:center}

/* ── Detection Log ── */
.log-panel{width:100%;max-width:900px;background:#1a1a1a;
  border:1px solid #333;border-radius:10px;padding:20px;
  display:flex;flex-direction:column;gap:12px}
.log-panel h3{font-size:.85rem;text-transform:uppercase;letter-spacing:2px;color:#aaa;text-align:center}
.log-table-wrap{max-height:300px;overflow-y:auto;border-radius:6px;
  border:1px solid #2a2a2a}
.log-table-wrap::-webkit-scrollbar{width:6px}
.log-table-wrap::-webkit-scrollbar-track{background:#222}
.log-table-wrap::-webkit-scrollbar-thumb{background:#444;border-radius:4px}
table.det-log{width:100%;border-collapse:collapse;font-size:.85rem}
table.det-log th{background:#222;color:#888;font-weight:600;
  text-transform:uppercase;letter-spacing:1px;padding:8px 12px;text-align:left}
table.det-log td{padding:8px 12px;border-top:1px solid #222;color:#ddd}
table.det-log tr:hover td{background:#232323}
.badge-seg{background:#1e3a5f;color:#64b5f6;border-radius:4px;padding:2px 7px;font-size:.78rem}
.badge-dist{background:#1a3a1a;color:#81c784;border-radius:4px;padding:2px 7px;font-size:.78rem}
</style>
</head>
<body>
<h1>Person Detection & AutoNav System</h1>

<!-- Top row: feeds + clip panel side by side -->
<div class="top-row">

  <!-- Left: feeds + audio controls stacked -->
  <div class="feeds">
    <div class="feeds-inner">
      <div class="feed-box">
        <h2 class="live-lbl">&#9679; Live Feed</h2>
        <img src="/raw" id="rawFeed">
        <button class="btn-grey" id="frontCamBtn" onclick="toggleCam('front',this)" style="width:100%;margin-top:4px">&#128249; Disable Front Cam</button>
      </div>
      <div class="feed-box">
        <h2 class="det-lbl">&#9679; Detection Feed</h2>
        <img src="/detection" id="detFeed">
      </div>
      <div class="feed-box" style="margin-top:8px">
        <h2 style="font-size:.85rem;text-transform:uppercase;letter-spacing:2px;color:#ab47bc">&#9679; Ceiling Cam</h2>
        <button class="btn-grey" id="ceilCamBtn" onclick="toggleCam('ceiling',this)" style="width:100%;margin-top:4px">&#128249; Disable Ceiling Cam</button>
        <button class="btn-blue" onclick="fetch('/api/swap_cameras').then(r=>r.json()).then(d=>setAudioSt(d.message))" style="width:100%;margin-top:4px">&#128257; Swap Top/Front Cameras</button>
      </div>
    </div>

    <!-- Automatic Audio Cues toggle -->
    <div class="toggle-row">
      <span class="toggle-name" id="aacLabel">Automatic Audio Cues</span>
      <label class="switch">
        <input type="checkbox" id="aacToggle" onchange="toggleAAC(this)">
        <span class="slider"></span>
      </label>
      <span class="toggle-sub" id="aacSub">off</span>
    </div>

    <!-- External Audio toggle -->
    <div class="toggle-row">
      <span class="toggle-name" id="extAudioLabel">External Audio</span>
      <label class="switch">
        <input type="checkbox" id="extAudioToggle" onchange="toggleExtAudio(this)">
        <span class="slider"></span>
      </label>
      <span class="toggle-sub" id="extAudioSub">off</span>
    </div>

    <!-- Audio controls -->
    <div class="row">
      <button class="btn-blue" onclick="playRandom()">&#9654; Random Clip</button>
      <button class="btn-red"  onclick="stopAudio()">&#9632; Stop Audio</button>
      <button class="btn-grey" id="micBtn" onclick="toggleMic()">&#127908; Mic</button>
    </div>
    <!-- Random source toggle -->
    <div class="row" style="margin-top:4px">
      <span style="font-size:.82rem;color:#aaa">Random Source:</span>
      <button class="btn-grey" id="srcAutoBtn"  onclick="setRandomSource('auto')">&#127925; Audio Clips</button>
      <button class="btn-grey" id="srcVoiceBtn" onclick="setRandomSource('voicelines')">&#127908; Voice Lines</button>
    </div>
    <!-- Volume -->
    <div class="vol-row">
      <label>&#128266; Volume</label>
      <input type="range" class="vol-slider" id="volSlider" min="0" max="100" value="10"
             oninput="setVolume(this.value)">
      <span id="volVal">10%</span>
    </div>
    <div id="audioStatus"></div>
  </div>

  <!-- Right: clip list sidebar -->
  <div class="clip-panel">
    <h3>&#127925; Audio Clips</h3>
    <button class="accordion active">Auto (Random Pool)</button>
    <div class="panel" style="max-height: 400px;" id="auto-sounds"></div>

    <button class="accordion active">Voice Lines</button>
    <div class="panel" style="max-height: 400px;" id="voicelines-sounds"></div>

    <button class="accordion active">Manual Only</button>
    <div class="panel" style="max-height: 400px;" id="manual-sounds"></div>
  </div>

</div>

<!-- ── Robot panel ── -->
<div class="robot-panel">
  <h3>&#129302; Robot Control</h3>

  <div class="conn-row">
    <div class="dot" id="connDot"></div>
    <span id="connText">Not connected</span>
  </div>

  <div class="row">
    <button class="btn-grey" onclick="rConnect()">&#128268; Connect</button>
    <button class="btn-red" onclick="rEStop()">&#9940; E-STOP</button>
  </div>
  
  <!-- AutoNav controls -->
  <div class="row" style="margin-top: 15px;">
    <button class="btn-green" onclick="fetch('/api/start').then(r=>r.json()).then(d=>setRobotSt(d.message))">&#9654; Start AutoNav</button>
    <button class="btn-red" onclick="fetch('/api/stop').then(r=>r.json()).then(d=>setRobotSt(d.message))">&#9208; Stop AutoNav</button>
  </div>
  <div class="row" style="margin-top: 15px;">
    <button class="btn-blue" onclick="fetch('/api/temp').then(r=>r.json()).then(d=>setRobotSt('Pi Temp: ' + d.temp, 5000))">&#127777;&#65039; Check Pi Temp</button>
  </div>

  <div id="robotStatus"></div>
</div>

<!-- ── Hallway Position Panel ── -->
<div class="log-panel" style="max-width:560px">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <h3>&#127968; Hallway Position</h3>
    <button class="btn-grey" style="padding:4px 10px;font-size:0.8rem" onclick="loadNavStatus()">&#128260; Refresh</button>
  </div>
  <div style="font-size:1.2rem;color:#eee;text-align:center;margin:10px 0;font-weight:bold" id="curSegment">Hallway: Unknown | Light: 0</div>
</div>

<!-- ── Detection Log panel ── -->
<div class="log-panel">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <h3>&#128205; Person Detection Log</h3>
    <button class="btn-grey" style="padding:4px 10px;font-size:0.8rem" onclick="loadDetectionLog()">&#128260; Refresh</button>
  </div>
  <div class="log-table-wrap">
    <table class="det-log" id="detLog">
      <thead><tr><th>#</th><th>Date and Time</th><th>People</th><th>Hallway</th><th>Light No.</th></tr></thead>
      <tbody id="detLogBody"><tr><td colspan="5" style="color:#555;text-align:center">No detections yet.</td></tr></tbody>
    </table>
  </div>
  <div style="display:flex;gap:10px;justify-content:center">
    <span style="font-size:.78rem;color:#555" id="detLogCount"></span>
  </div>
</div>

<script>
// ── Helpers ──────────────────────────────────────────────────────────────────
function setAudioSt(msg,d=3000){
  const e=document.getElementById("audioStatus");
  e.innerText=msg; if(d) setTimeout(()=>e.innerText="",d);
}
function setRobotSt(msg,d=3000){
  const e=document.getElementById("robotStatus");
  e.innerText=msg; if(d) setTimeout(()=>e.innerText="",d);
}

// ── Volume ────────────────────────────────────────────────────────────────────
function setVolume(val){
  document.getElementById("volVal").innerText = val + "%";
  fetch("/volume/"+val);
}
fetch("/volume/10");

// ── Detection Log ─────────────────────────────────────────────────────────────
function loadDetectionLog(){
  fetch("/api/detections").then(r=>r.json()).then(data=>{
    const body = document.getElementById("detLogBody");
    const count = document.getElementById("detLogCount");
    const rows = data.detections;
    if(!rows || rows.length === 0){
      body.innerHTML='<tr><td colspan="5" style="color:#555;text-align:center">No detections yet.</td></tr>';
      count.innerText = "";
      return;
    }
    count.innerText = rows.length + " total detection event" + (rows.length !== 1 ? "s" : "");
    body.innerHTML = rows.slice().reverse().map((r, i) =>
      `<tr>
        <td>${rows.length - i}</td>
        <td>${r.date_and_time}</td>
        <td><span class="badge-man">${r.count || 1}</span></td>
        <td><span class="badge-seg">HWY ${r.Hallway}</span></td>
        <td><span class="badge-dist">Light ${r.light_number}</span></td>
      </tr>`
    ).join("");
  });
}
loadDetectionLog();

// ── Nav Status ────────────────────────────────────────────────
function loadNavStatus(){
  fetch("/api/nav_status").then(r=>r.json()).then(d=>{
    document.getElementById("curSegment").innerText = "Hallway: " + d.hallway + " | Light: " + d.lights;
  });
}
loadNavStatus();

// ── AAC toggle ────────────────────────────────────────────────────────────────
function toggleAAC(cb){
  const on=cb.checked;
  fetch("/detectr/"+(on?"on":"off")).then(r=>r.json()).then(d=>{
    document.getElementById("aacLabel").className="toggle-name"+(on?" on":"");
    document.getElementById("aacSub").className  ="toggle-sub" +(on?" on":"");
    document.getElementById("aacSub").innerText  = on?"on — listening for people":"off";
    setAudioSt(d.message);
  });
}

// ── Audio ─────────────────────────────────────────────────────────────────────
function playRandom(){fetch("/play").then(r=>r.json()).then(d=>setAudioSt(d.message));}
function stopAudio(){
  document.querySelectorAll(".clip-btn").forEach(b=>b.classList.remove("playing"));
  fetch("/stop").then(r=>r.json()).then(d=>setAudioSt(d.message));
}
function playClip(name,folder,btn){
  document.querySelectorAll(".clip-btn").forEach(b=>b.classList.remove("playing"));
  btn.classList.add("playing");
  fetch("/play/"+encodeURIComponent(folder)+"/"+encodeURIComponent(name))
    .then(r=>r.json()).then(d=>setAudioSt(d.message));
}

// ── Random source toggle ──────────────────────────────────────────────────────
function setRandomSource(src){
  fetch("/api/set_random_source/"+src).then(r=>r.json()).then(d=>{
    setAudioSt(d.message);
    document.getElementById("srcAutoBtn") .classList.toggle("btn-blue", src==="auto");
    document.getElementById("srcAutoBtn") .classList.toggle("btn-grey", src!=="auto");
    document.getElementById("srcVoiceBtn").classList.toggle("btn-blue", src==="voicelines");
    document.getElementById("srcVoiceBtn").classList.toggle("btn-grey", src!=="voicelines");
  });
}
// Highlight the default source on load
setRandomSource("auto");

function loadClips(){
  fetch("/clips").then(r=>r.json()).then(data=>{
    const autoDiv  =document.getElementById("auto-sounds");
    const manDiv   =document.getElementById("manual-sounds");
    const voiceDiv =document.getElementById("voicelines-sounds");
    const auto  =data.clips.filter(c=>c.folder==="auto");
    const man   =data.clips.filter(c=>c.folder==="manual");
    const voice =data.clips.filter(c=>c.folder==="voicelines");
    
    autoDiv.innerHTML="";
    manDiv.innerHTML="";
    voiceDiv.innerHTML="";
    
    if(auto.length){
      auto.forEach(c=>autoDiv.appendChild(mkClipBtn(c)));
    } else {
      autoDiv.innerHTML='<div class="no-clips">No clips found</div>';
    }
    
    if(voice.length){
      voice.forEach(c=>voiceDiv.appendChild(mkClipBtn(c)));
    } else {
      voiceDiv.innerHTML='<div class="no-clips">No clips found</div>';
    }
    
    if(man.length){
      man.forEach(c=>manDiv.appendChild(mkClipBtn(c)));
    } else {
      manDiv.innerHTML='<div class="no-clips">No clips found</div>';
    }
    
    // Accordion logic
    var acc = document.getElementsByClassName("accordion");
    for (var i = 0; i < acc.length; i++) {
        // Prevent multiple bindings if called twice
        if (!acc[i].hasAttribute("data-bound")) {
            acc[i].setAttribute("data-bound", "true");
            acc[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var panel = this.nextElementSibling;
                if (panel.style.maxHeight) {
                    panel.style.maxHeight = null;
                } else {
                    panel.style.maxHeight = "400px";
                }
            });
        }
    }
  });
}
function mkClipBtn(c){
  const b=document.createElement("button");
  b.className="clip-btn";
  const ext=c.name.split(".").pop().toLowerCase();
  const manBadge   =c.folder==="manual"     ?'<span class="badge-man">[MANUAL]</span>':"";
  const voiceBadge =c.folder==="voicelines" ?'<span class="badge-mp3">[VOICE]</span>':"";
  b.innerHTML=`${manBadge}${voiceBadge}<span class="badge-${ext}">[${ext.toUpperCase()}]</span>&#9654;  ${c.name}`;
  b.onclick=()=>playClip(c.name,c.folder,b);
  return b;
}
loadClips();

// ── External Audio (webcam mic → browser) ─────────────────────────────────────
let extAudioOn=false,extAudioSock=null,audioCtx=null,nextPlayTime=0;
const EXT_SR=22050;

function toggleExtAudio(cb){ cb.checked?startExtAudio():stopExtAudio(); }

function startExtAudio(){
  audioCtx=new (window.AudioContext||window.webkitAudioContext)({sampleRate:EXT_SR});
  nextPlayTime=audioCtx.currentTime+0.1;
  extAudioSock=new WebSocket("wss://"+location.host+"/ext_audio");
  extAudioSock.binaryType="arraybuffer";
  extAudioSock.onopen=()=>{
    extAudioOn=true;
    document.getElementById("extAudioLabel").className="toggle-name on";
    document.getElementById("extAudioSub").className="toggle-sub on";
    document.getElementById("extAudioSub").innerText="on — streaming webcam mic";
    setAudioSt("External audio streaming...",0);
  };
  extAudioSock.onmessage=e=>{
    if(!audioCtx) return;
    const raw=new Int16Array(e.data);
    const f32=new Float32Array(raw.length);
    for(let i=0;i<raw.length;i++) f32[i]=raw[i]/32768.0;
    const buf=audioCtx.createBuffer(1,f32.length,EXT_SR);
    buf.getChannelData(0).set(f32);
    const src=audioCtx.createBufferSource();
    src.buffer=buf; src.connect(audioCtx.destination);
    const t=Math.max(audioCtx.currentTime,nextPlayTime);
    src.start(t); nextPlayTime=t+buf.duration;
  };
  extAudioSock.onerror=()=>stopExtAudio();
  extAudioSock.onclose=()=>{ if(extAudioOn) stopExtAudio(); };
}

function stopExtAudio(){
  if(extAudioSock){extAudioSock.close();extAudioSock=null;}
  if(audioCtx){audioCtx.close();audioCtx=null;}
  extAudioOn=false;
  document.getElementById("extAudioLabel").className="toggle-name";
  document.getElementById("extAudioSub").className="toggle-sub";
  document.getElementById("extAudioSub").innerText="off";
  document.getElementById("extAudioToggle").checked=false;
  setAudioSt("External audio off.");
}

// ── Mic ───────────────────────────────────────────────────────────────────────
let micOn=false,recorder=null,micSock=null;
function toggleMic(){ micOn?stopMic():startMic(); }
function startMic(){
  navigator.mediaDevices.getUserMedia({audio:true,video:false}).then(stream=>{
    micSock=new WebSocket("wss://"+location.host+"/mic");
    micSock.binaryType="arraybuffer";
    micSock.onopen=()=>{
      recorder=new MediaRecorder(stream,{mimeType:"audio/webm;codecs=opus"});
      recorder.ondataavailable=e=>{
        if(e.data.size>0&&micSock.readyState===WebSocket.OPEN) micSock.send(e.data);
      };
      recorder.start(250); micOn=true;
      document.getElementById("micBtn").classList.add("on");
      document.getElementById("micBtn").textContent="🎙 Mic ON";
      setAudioSt("Mic transmitting...",0);
    };
    micSock.onerror=()=>{setAudioSt("Mic error.");stopMic();};
  }).catch(e=>setAudioSt("Mic permission denied: "+e.message));
}
function stopMic(){
  if(recorder){recorder.stop();recorder=null;}
  if(micSock){micSock.close();micSock=null;}
  micOn=false;
  document.getElementById("micBtn").classList.remove("on");
  document.getElementById("micBtn").textContent="🎙 Mic";
  setAudioSt("Mic off.");
}

// ── Robot ─────────────────────────────────────────────────────────────────────
function setDot(state){
  const dot=document.getElementById("connDot");
  const txt=document.getElementById("connText");
  dot.className="dot "+(state||"");
  txt.innerText=state==="conn"?"Connected — AutoNav ready":
                state==="pi"  ?"Connected — Active":
                "Not connected";
}

function rConnect(){
  fetch("/robot/connect").then(r=>r.json()).then(d=>{
    setRobotSt(d.message); if(d.ok) setDot("conn");
  });
}
function rEStop(){
  fetch("/api/stop").then(r=>r.json()).then(d=>{
    setRobotSt(d.message); setDot("pi");
  });
}
</script>
</body>
</html>'''

# ─── Startup ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(AUDIO_FOLDER,      exist_ok=True)
    os.makedirs(MANUAL_FOLDER,     exist_ok=True)
    os.makedirs(VOICELINES_FOLDER, exist_ok=True)
    threading.Thread(target=capture_frames, daemon=True).start()
    threading.Thread(target=capture_ceiling_frames, daemon=True).start()
    threading.Thread(target=navigation_loop, daemon=True).start()

    print("\n Person Detection & AutoNav running!")
    print(f" Auto clips:      {AUDIO_FOLDER}")
    print(f" Manual clips:    {MANUAL_FOLDER}")
    print(f" Voice lines:     {VOICELINES_FOLDER}")
    print(f" Robot port:      {ROBOT_PORT}")
    print(f" Ext mic card:    {EXT_MIC_CARD} (change EXT_MIC_CARD if wrong)")
    print(f"\n Open https://<your-pi-ip>:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False,
            ssl_context=('/home/bonnerbot/cert.pem', '/home/bonnerbot/key.pem'))
