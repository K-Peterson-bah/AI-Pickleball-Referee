import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

# == define ball tracking class == #
class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate().bfill()
        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        return ball_dict

    def draw_bbox(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.putText(frame, f"Ball", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

# === Load Model === #
ball_model = BallTracker("./runs/detect/train/weights/best.pt")
court_model = YOLO("runs/pose/train9/weights/best.pt")

# === input video === #
video_path ="./videos/vid4.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
birdseye_size = (400,900)

#resize for overlay
overlay_width = input_width // 3
overlay_height = int(birdseye_size[1] * (overlay_width / birdseye_size[0]))

output = cv2.VideoWriter("output.mp4", 
                         cv2.VideoWriter_fourcc(*'mp4v'),
                           fps, (input_width, input_height))

# === Define reference destination points === #
# adjust as needed
dst_pts = np.array([
    [0,0],  # top left
    [200, 0], # top middle
    [400, 0], # top right
    [0, 300], # top left kitchen
    [200, 300], #top middle kitchen
    [400, 400], #top right kitchen
    [0, 580], # botttom left kitchen
    [200, 580], #bottom middle kitchen
    [400, 580], #bottom right kitchen
    [0, 880], #bottom left
    [200, 880], #bottom midle
    [400, 880] #bottom right
], dtype=np.float32)

# === Define line connections === #
connect_pairs =[
    (0,1), (1, 2), (0,3),
    (1, 4), (2, 5), (3,4),
    (4,5), (3,6), (5, 8),
    (6, 7), (7,8), (6,9),
    (7, 10), (8,11), (9, 10),
    (10,11)
]

# === store ball detections to interpolate after processing all frames === #
frames = []
ball_detections = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    ball_dict = ball_model.detect_frame(frame)
    ball_detections.append(ball_dict)
    
cap.release()
# === Step 2: Interpolate Ball Trajectory === #
ball_detections_interp = ball_model.interpolate_ball_positions(ball_detections)


for frame_idx, frame in enumerate(frames):
    # == run Yolo inference == #
    results = court_model(frame, conf=0.9)
    keypoints = results[0].keypoints
    
    #prepare blank canvasd for homography view
    canvas = np.zeros((birdseye_size[1], birdseye_size[0], 3), dtype=np.uint8)
    H = None

    # == ensure prediction exists == #
    if keypoints is not None and keypoints.shape[1] == 12:
        #get keypoints in pixel space
        kpts = keypoints.xy[0].cpu().numpy().astype(np.float32)

        #compute homography
        if kpts.shape == dst_pts.shape:
            H, _ = cv2.findHomography(kpts, dst_pts, method=cv2.RANSAC)

            if H is not None:
                #project keypoints to birds-eye view
                kpts_homo = cv2.perspectiveTransform(kpts[None, :, :], H)[0]

                #draw keypoints
                for pt in kpts_homo:
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(canvas, (x,y), 5, (0, 255, 0), -1)

                #draw connecting lines
                for i, j in connect_pairs:
                    pt1 = tuple(np.round(kpts_homo[i]).astype(int))
                    pt2 = tuple(np.round(kpts_homo[j]).astype(int))
                    cv2.line(canvas, pt1, pt2, (255, 255, 255), 2)

    # == ball interpolation == #
    ball_dict = ball_detections_interp[frame_idx]
    if 1 in ball_dict:
        bbox = ball_dict[1]
        ball_model.draw_bbox(frame, bbox)

        # === Ball on Homography === #
        if H is not None and 1 in ball_dict:
            x1, y1, x2, y2 = ball_dict[1]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ball_pt = np.array([[[cx, cy]]], dtype=np.float32)
            ball_proj = cv2.perspectiveTransform(ball_pt, H)[0][0]
            cv2.circle(canvas, tuple(ball_proj.astype(int)), 6, (0, 0, 255), -1)
    '''
    # == resize canvas to overlay size == #
    aspect_ratio = birdseye_size[1] / birdseye_size[0]
    overlay_width = input_width // 3
    overlay_height = min(input_height, int(overlay_width * aspect_ratio))

    canvas_overlay = cv2.resize(canvas, (overlay_width, overlay_height))

    # == overlay on top-right of original frame == #
    x_offset = input_width - overlay_width
    y_offset = 0

    #created transparent overlay
    alpha = 0.5  # transparency factor, 0.0 = fully transparent, 1.0 = fully opaque

    # Extract the ROI from the original frame
    roi = frame[y_offset:y_offset + overlay_height, x_offset:x_offset + overlay_width]

    # Blend the ROI and overlay using alpha transparency
    blended = cv2.addWeighted(roi, 1 - alpha, canvas_overlay, alpha, 0)

    # Put blended result back into original frame
    frame[y_offset:y_offset + overlay_height, x_offset:x_offset + overlay_width] = blended
    '''
    # === Composite Overlay: Transparent BG, Opaque Ball/Court === #
    overlay_height, overlay_width = canvas.shape[:2]
    x_offset = input_width - overlay_width
    y_offset = 0

    canvas_overlay = frame[y_offset:y_offset + overlay_height, x_offset:x_offset + overlay_width].copy()
    blended_bg = cv2.addWeighted(canvas_overlay, 0.6, canvas, 0.4, 0)

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    bg_part = cv2.bitwise_and(blended_bg, blended_bg, mask=mask_inv)
    fg_part = cv2.bitwise_and(canvas, canvas, mask=mask)
    combined = cv2.add(bg_part, fg_part)

    frame[y_offset:y_offset + overlay_height, x_offset:x_offset + overlay_width] = combined
    output.write(frame)
    frame_idx += 1

cap.release()
output.release()
cv2.destroyAllWindows()