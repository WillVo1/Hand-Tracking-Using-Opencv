import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import urllib.request
import os
import numpy as np
from scipy.interpolate import splprep, splev
from collections import deque

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )

detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2
    )
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
CAM_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
CAM_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[Camera] Resolution: {CAM_W}x{CAM_H}")

# Load back.png overlay image
back_img = cv2.imread("back.png", cv2.IMREAD_UNCHANGED)
if back_img is None:
    print("Warning: back.png not found!")
    back_img_resized = None
else:
    print(f"[Overlay] Loaded back.png: {back_img.shape}")
    # Resize overlay - 3x bigger than half screen
    bh, bw = back_img.shape[:2]
    scale = min((CAM_W * 0.5) / bw, (CAM_H * 0.5) / bh) * 4.5
    new_w = int(bw * scale)
    new_h = int(bh * scale)
    back_img_resized = cv2.resize(back_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(f"[Overlay] Resized to: {back_img_resized.shape}")

pTime = 0
trail_points = []
frozen_trail = []
recent_positions = deque(maxlen=10)
still_since = None
drawing = False
done = False
done_time = None
last_draw_pos = None
frozen_frame = None

while True:
    success, img = cap.read()
    if not success:
        break

    # If done and 3 seconds has passed, show frozen frame
    if done and done_time is not None and (time.time() - done_time) >= 3.0:
        if frozen_frame is None:
            # Need to process one more frame to capture the final state with drawing
            pass
        else:
            cv2.imshow("Image", frozen_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reset
                trail_points = []
                frozen_trail = []
                done = False
                done_time = None
                drawing = False
                still_since = None
                last_draw_pos = None
                recent_positions.clear()
                frozen_frame = None
            continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB))

    # Always draw frozen trail if done (even without hand detection)
    if done and len(frozen_trail) >= 4:
        seen = set()
        deduped = [p for p in frozen_trail if not (tuple(p) in seen or seen.add(tuple(p)))]
        pts = np.array(deduped, dtype=np.float32)
        if len(pts) >= 4:
            try:
                tck, u = splprep([pts[:, 0], pts[:, 1]], s=0, k=min(3, len(pts)-1))
                fine = np.linspace(0, 1, 300)
                sx, sy = splev(fine, tck)
                spline_pts = np.array(list(zip(sx.astype(int), sy.astype(int))))
                cv2.polylines(img, [spline_pts.reshape(-1, 1, 2)], False, (0, 0, 255), 2)
            except:
                pass

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            h, w, _ = img.shape
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            for id, (cx, cy) in enumerate(points):
                if id == 0:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                else:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if len(points) > 8:
                cx, cy = points[8]
                recent_positions.append((cx, cy))

                if len(recent_positions) == 10:
                    avg_x = sum(p[0] for p in recent_positions) / 10
                    avg_y = sum(p[1] for p in recent_positions) / 10
                    spread = max(((p[0]-avg_x)**2 + (p[1]-avg_y)**2)**0.5 for p in recent_positions)
                    is_still = spread < 15

                    if not done:
                        if is_still:
                            if still_since is None:
                                still_since = time.time()
                            elapsed = time.time() - still_since
                            if elapsed >= 1:
                                if not drawing:
                                    drawing = True
                                    still_since = None
                                elif drawing:
                                    done = True
                                    done_time = time.time()
                                    drawing = False
                                    frozen_trail = trail_points.copy()
                                    still_since = None
                                    recent_positions.clear()

                        else:
                            still_since = None
                            if drawing:
                                if last_draw_pos is None or ((cx - last_draw_pos[0])**2 + (cy - last_draw_pos[1])**2)**0.5 > 15:
                                    trail_points.append((cx, cy))
                                    last_draw_pos = (cx, cy)
                                    if len(trail_points) > 50:
                                        trail_points.pop(0)

            # draw spline from live trail or frozen trail
            draw_pts = frozen_trail if done else (trail_points + [points[8]] if drawing else [])
            if len(draw_pts) >= 4:
                seen = set()
                deduped = [p for p in draw_pts if not (tuple(p) in seen or seen.add(tuple(p)))]
                pts = np.array(deduped, dtype=np.float32)
                if len(pts) >= 4:
                    try:
                        tck, u = splprep([pts[:, 0], pts[:, 1]], s=0, k=min(3, len(pts)-1))
                        fine = np.linspace(0, 1, 300)
                        sx, sy = splev(fine, tck)
                        spline_pts = np.array(list(zip(sx.astype(int), sy.astype(int))))
                        cv2.polylines(img, [spline_pts.reshape(-1, 1, 2)], False, (0, 0, 255), 2)
                    except:
                        pass

            for a, b in HAND_CONNECTIONS:
                cv2.line(img, points[a], points[b], (0, 255, 0), 2)

    # Overlay back.png in center when done
    if done and back_img_resized is not None:
        h, w = img.shape[:2]
        bh, bw = back_img_resized.shape[:2]
        
        # Calculate center position
        x = (w - bw) // 2
        y = (h - bh) // 2
        
        # Calculate clipping boundaries
        src_x1 = max(0, -x)
        src_y1 = max(0, -y)
        src_x2 = min(bw, w - x)
        src_y2 = min(bh, h - y)
        
        dst_x1 = max(0, x)
        dst_y1 = max(0, y)
        dst_x2 = min(w, x + bw)
        dst_y2 = min(h, y + bh)
        
        # Only overlay the visible portion
        if src_x2 > src_x1 and src_y2 > src_y1:
            overlay_section = back_img_resized[src_y1:src_y2, src_x1:src_x2]
            
            # Overlay with alpha channel if available
            if back_img_resized.shape[2] == 4:  # Has alpha channel
                alpha = overlay_section[:, :, 3:4] / 255.0
                for c in range(3):
                    img[dst_y1:dst_y2, dst_x1:dst_x2, c] = (
                        alpha[:, :, 0] * overlay_section[:, :, c] + 
                        (1 - alpha[:, :, 0]) * img[dst_y1:dst_y2, dst_x1:dst_x2, c]
                    )
            else:
                img[dst_y1:dst_y2, dst_x1:dst_x2] = overlay_section[:, :, :3]

    # Display status text (even without hand detection)
    if done:
        status = "DONE - 'r':reset"
    elif drawing:
        status = f"STOP IN: {max(0, 1-(time.time()-still_since)):.1f}s" if still_since else "DRAWING"
    else:
        status = f"START IN: {max(0, 1-(time.time()-still_since)):.1f}s" if still_since else "HOLD STILL 1s"
    cv2.putText(img, status, (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    # Capture frozen frame after 3 second delay
    if done and done_time is not None and (time.time() - done_time) >= 3.0 and frozen_frame is None:
        frozen_frame = img.copy()
    
    cv2.imshow("Image", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):  # Reset
        trail_points = []
        frozen_trail = []
        done = False
        done_time = None
        drawing = False
        still_since = None
        last_draw_pos = None
        recent_positions.clear()
        frozen_frame = None

cap.release()
cv2.destroyAllWindows()
