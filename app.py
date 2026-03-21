import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import urllib.request
import os
from dotenv import load_dotenv
import numpy as np
from scipy.interpolate import splprep, splev
from collections import deque
from real_scoliosis_ai import RealScoliosisAI

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
pTime = 0
trail_points = []
frozen_trail = []
recent_positions = deque(maxlen=10)
still_since = None
drawing = False
done = False
last_draw_pos = None

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
real_ai = RealScoliosisAI(api_key)
ai_corrected_coords = []
show_ai_results = False
ai_processing = False

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB))

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
                                    recent_positions.clear()
                                elif drawing:
                                    done = True
                                    drawing = False
                                    frozen_trail = trail_points.copy()
                                    still_since = None
                                    recent_positions.clear()
                                    
                                    # Process with REAL AI when drawing is complete
                                    if len(frozen_trail) > 3:
                                        ai_processing = True
                                        import threading
                                        snapshot = img.copy()  # capture frame NOW before loop advances
                                        trail_snapshot = frozen_trail.copy()
                                        def ai_analysis(frame=snapshot, trail=trail_snapshot):
                                            global ai_corrected_coords, ai_processing, show_ai_results
                                            coords, _ = real_ai.analyze_and_correct(trail, frame)
                                            ai_corrected_coords = coords
                                            ai_processing = False
                                            show_ai_results = True
                                        threading.Thread(target=ai_analysis, daemon=True).start()
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

            if done:
                if ai_processing:
                    status = "🤖 AI ANALYZING SPINE... Please wait"
                else:
                    status = "DONE - 'r':reset 'a':toggle AI results"
            elif drawing:
                status = f"STOP IN: {max(0, 1-(time.time()-still_since)):.1f}s" if still_since else "DRAWING"
            else:
                status = f"START IN: {max(0, 1-(time.time()-still_since)):.1f}s" if still_since else "HOLD STILL 1s"
            cv2.putText(img, status, (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
            
            # Show AI corrected spine
            if show_ai_results and ai_corrected_coords and not ai_processing:
                if len(ai_corrected_coords) > 1:
                    pts = np.array(ai_corrected_coords, dtype=np.int32)
                    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (0, 255, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):  # Reset
        trail_points = []
        frozen_trail = []
        ai_corrected_coords = []
        done = False
        drawing = False
        show_ai_results = False
        ai_processing = False
        still_since = None
        recent_positions.clear()
    elif key == ord('a'):  # Toggle AI results
        show_ai_results = not show_ai_results

cap.release()
cv2.destroyAllWindows()
