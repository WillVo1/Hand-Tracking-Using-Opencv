import cv2
import numpy as np
import base64
import json
import re
import requests
from typing import List, Tuple, Dict, Any


class RealScoliosisAI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def _encode_image(self, image: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buffer).decode('utf-8')

    def _draw_user_trail(self, image: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
        """Render the user's drawn spline onto the image in bright red so the model sees it clearly."""
        overlay = image.copy()
        if len(points) > 1:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], False, (0, 0, 255), 4)
            # Mark start and end
            cv2.circle(overlay, points[0], 8, (0, 255, 255), -1)
            cv2.circle(overlay, points[-1], 8, (255, 0, 0), -1)
        return overlay

    def analyze_and_correct(self, user_drawing: List[Tuple[int, int]], image: np.ndarray) -> Tuple[List[Tuple[int, int]], Dict]:
        if not user_drawing:
            return [], {"error": "No drawing provided"}

        h, w = image.shape[:2]

        # Build annotated image
        annotated = self._draw_user_trail(image, user_drawing)
        b64 = self._encode_image(annotated)

        # Serialize the drawn points as plain text so the model has exact pixel coords
        points_str = ", ".join(f"({x},{y})" for x, y in user_drawing)

        prompt = f"""You are a spine anatomy expert. A user has drawn a rough curve on an image attempting to trace a spine.

IMAGE RESOLUTION: {w}x{h} pixels
USER'S DRAWN POINTS (pixel coordinates, top to bottom):
{points_str}

Ignore how accurate the user's drawing is. Look at the image and identify the true spine centerline yourself.
Return exactly {len(user_drawing)} evenly spaced points along the TRUE spine centerline from top to bottom, fully corrected to match the actual anatomy visible in the image.
Do NOT follow the user's drawing — use it only to understand the general region of interest.

Respond ONLY with this JSON, no markdown, no explanation:
{{"corrected_spine_points": [[x1,y1],[x2,y2],...]}}"""

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.0
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=45)
            response.raise_for_status()
            resp_json = response.json()
            choice = resp_json['choices'][0]
            content = choice['message'].get('content') or ''
            if not content:
                print(f"AI returned empty content. Finish reason: {choice.get('finish_reason')}")
                return user_drawing, {}
            content = content.strip()

            # Strip markdown code fences if present
            content = re.sub(r'^```(?:json)?\s*', '', content)
            content = re.sub(r'\s*```$', '', content)

            result = json.loads(content)

            raw_pts = result.get("corrected_spine_points", [])
            corrected = []
            for p in raw_pts:
                if isinstance(p, (list, tuple)) and len(p) == 2:
                    x, y = int(p[0]), int(p[1])
                    corrected.append((max(0, min(x, w-1)), max(0, min(y, h-1))))

            return corrected if corrected else user_drawing, {}

        except requests.exceptions.RequestException as e:
            print(f"API error: {e}")
            return user_drawing, {}
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Parse error: {e}")
            return user_drawing, {}
