import cv2
from board_warper import warp_chessboard, get_closest_tile
import numpy as np
def read_first_frame(folder, path):
    root_path = "cu-chess-detection-2025/Chess Detection Competition/"
    output_dir = "first-frame"
    video_path = root_path + folder + "/" + path
    cap = cv2.VideoCapture(video_path)
    frame = None
    while True:
        ret, frame = cap.read()
        cv2.imwrite(f"{output_dir}/{path.split('.')[0]}.jpg", frame)

        break

    cap.release()
    img_path = f"{output_dir}/{path.split('.')[0]}.jpg"
    return img_path, frame
def hand_mask_skin(frame):
    img = cv2.GaussianBlur(frame, (5, 5), 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # HSV mask
    lower_hsv = np.array([0, 30, 60])
    upper_hsv = np.array([25, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # YCrCb mask
    lower_ycc = np.array([0, 135, 85])
    upper_ycc = np.array([255, 180, 135])
    mask_ycc = cv2.inRange(ycrcb, lower_ycc, upper_ycc)

    mask = cv2.bitwise_or(mask_hsv, mask_ycc)
    
    # --- Light cleanup ---
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small gaps
    #Might need to remove
    # mask = cv2.GaussianBlur(mask, (7, 7), 0)
    # _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask
def draw_hand_mask(frame, mask, color=(0, 0, 255), alpha=0.5):
    """
    Draws a colored transparent overlay of the mask on the original frame.

    Parameters:
        frame (np.ndarray): Original BGR image.
        mask (np.ndarray): Binary mask (0/255).
        color (tuple): BGR color for overlay. Default red.
        alpha (float): Transparency factor (0–1).

    Returns:
        np.ndarray: Frame with mask overlay.
    """
    # Ensure mask is 3-channel
    mask_color = np.zeros_like(frame)
    mask_color[mask > 0] = color

    # Blend images (only where mask present)
    overlay = cv2.addWeighted(mask_color, alpha, frame, 1 - alpha, 0)

    # Use mask to selectively update pixels
    output = frame.copy()
    output[mask > 0] = overlay[mask > 0]

    return output

# def compute_edge_mean(center, frame, tile_size=103, thresh=25):
#     """
#     center: (cx, cy) tile center coordinates
#     frame: image (BGR)
#     tile_size: width/height of each tile
#     thresh: pixel threshold to decide if something exists
#     """
#     h, w = frame.shape[:2]
#     cx, cy = center

#     half = tile_size // 2

#     # Compute tile bounds (clamped to image edges)
#     x1 = max(cy - half, 0)  # frame[y][x] so watch order
#     y1 = max(cx - half, 0)
#     x2 = min(cy + half, w)
#     y2 = min(cx + half, h)

#     # Extract ROI
#     tile_roi = frame[y1:y2, x1:x2]
#     color_mean = np.mean(tile_roi)
#     #For Debugging
#     # plt.imshow(cv2.cvtColor(tile_roi, cv2.COLOR_BGR2RGB))
#     # plt.show()

#     # Convert to grayscale
#     gray = cv2.cvtColor(tile_roi, cv2.COLOR_BGR2GRAY)

#     # Detect piece by intensity variation
#     # Compute standard deviation: high means structure → piece exists
#     edges = cv2.Canny(gray, 50, 150)
    
#     # plt.imshow(edges, cmap="gray")
#     # plt.show()
#     # print(edges.mean())
#     return edges.mean(), color_mean

# def validate_move(uncropped_before, uncropped_after, centers, tile):
#     c = centers[tile]
#     before_edge_mean, before_color_mean = compute_edge_mean(c, uncropped_before)
#     after_edge_mean, after_color_mean = compute_edge_mean(c, uncropped_after)
    
#     edge_diff = abs(before_edge_mean-after_edge_mean)
#     color_diff = abs(before_color_mean-after_color_mean)

#     if edge_diff < 1.5 and color_diff < 15:
#         return False
#     else:
#         return True
def compute_edge_mean(center, frame, tile_size=103, thresh=25):
    """
    center: (cx, cy) tile center coordinates
    frame: image (BGR)
    tile_size: width/height of each tile
    thresh: pixel threshold to decide if something exists
    """
    h, w = frame.shape[:2]
    cx, cy = center

    half = tile_size // 2

    # Compute tile bounds (clamped to image edges)
    x1 = max(cy - half, 0)  # frame[y][x] so watch order
    y1 = max(cx - half, 0)
    x2 = min(cy + half, w)
    y2 = min(cx + half, h)

    # Extract ROI
    tile_roi = frame[y1:y2, x1:x2]
    tile_hsv = cv2.cvtColor(tile_roi, cv2.COLOR_BGR2HSV)
    hue_mean = np.mean(tile_hsv[:, :, 0])
    sat_mean = np.mean(tile_hsv[:, :, 1])
    #For Debugging
    # plt.imshow(cv2.cvtColor(tile_roi, cv2.COLOR_BGR2RGB))
    # plt.show()

    # Convert to grayscale
    gray = cv2.cvtColor(tile_roi, cv2.COLOR_BGR2GRAY)

    # Detect piece by intensity variation
    # Compute standard deviation: high means structure → piece exists
    edges = cv2.Canny(gray, 50, 150)
    
    # plt.imshow(edges, cmap="gray")
    # plt.show()
    # print(edges.mean())
    return edges.mean(), hue_mean, sat_mean

def validate_move(uncropped_before, uncropped_after, centers, tile):
    c = centers[tile]
    before_edge_mean, before_hue, before_sat = compute_edge_mean(c, uncropped_before)
    after_edge_mean, after_hue, after_sat = compute_edge_mean(c, uncropped_after)
    
    edge_diff = abs(before_edge_mean - after_edge_mean)
    hue_diff = abs(before_hue - after_hue)
    sat_diff = abs(before_sat - after_sat)
    
    # Handle hue wraparound (hue is circular: 0° = 180°)
    hue_diff = min(hue_diff, 180 - hue_diff)
    
    # Adjust thresholds as needed
    # print(hue_diff)
    # print("edge_diff", edge_diff)
    if edge_diff < 2 and hue_diff < 15:
        return False
    else:
        return True


def find_move(before, after, centers, warped=True, ref=None, MIN_AREA=920, MAX_AREA=12000, crop = 90):
    if warped:
        warped_before = before.copy()
        warped_after = after.copy()
    else:
        warped_before = warp_chessboard(before, ref=ref)
        warped_after = warp_chessboard(after, ref=ref)
    
    h, w = warped_before.shape[:2]
    before_uncropped = warped_before.copy()
    after_uncropped = warped_after.copy()
    warped_before = warped_before[crop:h-crop, crop:w-crop].copy()
    warped_after = warped_after[crop:h-crop, crop:w-crop].copy()

    #Gaussian Blur to reduce noise
    warped_before = cv2.GaussianBlur(warped_before, (13,13), 0)
    warped_after = cv2.GaussianBlur(warped_after, (13,13), 0)

    diff = cv2.absdiff(warped_before, warped_after)

    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    #_, thresh = cv2.threshold(diff_gray, 15, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(
        diff_gray, 
        255, 
        cv2.ADAPTIVE_THRESH_MEAN_C,  # or cv2.ADAPTIVE_THRESH_MEAN_C
        cv2.THRESH_BINARY, 
        blockSize=401,  # Must be odd; adjust based on tile size
        C=-17  # Negative value to keep threshold around 15
    )
    # ----- Added erosion -----
    kernel = np.ones((7,7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # ------- Debug -----------
    # plt.imshow(diff_gray, cmap='gray')
    # plt.show()
    # plt.imshow(thresh, cmap='gray')
    # plt.show()
    if cv2.countNonZero(thresh) > 30000:
        return [], []
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnts = cnts[:2]    # only the 2 largest
    # -------------------------------------------------------

    detected_tiles = []
    areas = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        areas.append(area)

        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w//2 + crop, y + h//2 + crop

        tile = get_closest_tile(cx, cy, centers)
        if validate_move(before_uncropped, after_uncropped, centers, tile) == True:
            detected_tiles.append(tile)
    if len(detected_tiles) == 2:
        if detected_tiles[0] == detected_tiles[1]:
            detected_tiles = []
    return detected_tiles, areas



def extract_frames(video_path, ref=None, centers=None, crop = 90):
    folder = video_path.split("\\")[2]
    path = video_path.split("\\")[3]
    _, first_frame = read_first_frame(folder, path)
    
    cap = cv2.VideoCapture(video_path)

    extracted_frames = []
    
    current_frame = warp_chessboard(first_frame, ref=ref)
    extracted_frames.append(current_frame)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % 2 == 0:
            frame_index += 1
            continue
        frame = warp_chessboard(frame, ref=ref)

        h, w = frame.shape[:2]

        frame_cropped = frame[crop:h-crop, crop:w-crop].copy()

        hand_pixels = cv2.countNonZero(hand_mask_skin(frame_cropped))
        if hand_pixels > 2500:#From 4000
            continue

        move, _ = find_move(current_frame, frame, centers)
        if len(move) == 2:
            extracted_frames.append(frame)
            current_frame = frame.copy()
        frame_index += 1
    cap.release()
    return first_frame, extracted_frames



#----------------------- V2 ---------------------------------------
# def adjust_gamma(image, gamma=1.0):
#     # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255
#         for i in np.arange(0, 256)]).astype("uint8")
    
#     # Apply gamma correction using the lookup table
#     return cv2.LUT(image, table)
# def find_move(before, after, centers, warped=True, ref=None, MIN_AREA=700, MAX_AREA=12000, crop = 90):
#     if warped:
#         warped_before = before.copy()
#         warped_after = after.copy()
#     else:
#         warped_before = warp_chessboard(before, ref=ref)
#         warped_after = warp_chessboard(after, ref=ref)
    
#     h, w = warped_before.shape[:2]
#     before_uncropped = warped_before.copy()
#     after_uncropped = warped_after.copy()
#     warped_before = warped_before[crop:h-crop, crop:w-crop].copy()
#     warped_after = warped_after[crop:h-crop, crop:w-crop].copy()

#     diff = cv2.absdiff(warped_before, warped_after)

#     diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
#     thresh = cv2.adaptiveThreshold(
#             diff_gray, 
#             255, 
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#             cv2.THRESH_BINARY, 
#             blockSize=201,  
#             C=-10
#         )
#     #_, thresh = cv2.threshold(diff_gray, 15, 255, cv2.THRESH_BINARY)

#     # ----- Added erosion -----
#     kernel = np.ones((7,7), np.uint8)
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
#     #------- Debug -----------
#     # plt.imshow(diff_gray, cmap='gray')
#     # plt.show()
#     # plt.imshow(thresh, cmap='gray')
#     # plt.show()

#     cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
#     cnts = cnts[:2]    # only the 2 largest
#     # -------------------------------------------------------

#     detected_tiles = []
#     areas = []
#     for c in cnts:
        
#         area = cv2.contourArea(c)
#         #print(area)
#         if area < MIN_AREA or area > MAX_AREA:
#             continue
#         areas.append(area)

#         x, y, w, h = cv2.boundingRect(c)
#         cx, cy = x + w//2 + crop, y + h//2 + crop

#         tile = get_closest_tile(cx, cy, centers)
#         if validate_move(before_uncropped, after_uncropped, centers, tile) == True:
#             detected_tiles.append(tile)

#     return detected_tiles, areas