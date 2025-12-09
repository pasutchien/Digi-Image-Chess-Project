import cv2
import numpy as np

def get_lines(edges):
    lines = cv2.HoughLines(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=175
    )

    vertical = []
    horizontal = []
    
    for line in lines:
        rho, theta = line[0]
        
        if abs(np.sin(theta)) < 0.5:  # vertical-ish
            vertical.append((rho, theta))
        
        else:  # horizontal-ish
            horizontal.append((rho, theta))
    return vertical, horizontal

def intersect(rho1, theta1, rho2, theta2):
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])
    x, y = np.linalg.solve(A, b)
    return int(x), int(y)

def warp_chessboard(frame, board_size=920, ref=None, image_size=1024):
    '''Receives frame as BGR image'''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Blur (5x5)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # 2. Threshold 200
    _, thresh_img = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    # 3. Canny
    edges = cv2.Canny(thresh_img, 50, 150)
    
    vertical, horizontal = get_lines(edges)
    
    r_vertical = sorted(vertical, key=lambda x: abs(x[0]), reverse=True)[0]
    r_horizontal = sorted(horizontal, key=lambda x: abs(x[0]), reverse=True)[0]

    l_vertical = sorted(vertical, key=lambda x: abs(x[0]), reverse=False)[0]
    l_horizontal = sorted(horizontal, key=lambda x: abs(x[0]), reverse=False)[0]
    
    corner1 = intersect(l_vertical[0], l_vertical[1], l_horizontal[0], l_horizontal[1]) # top-left
    corner2 = intersect(r_vertical[0], r_vertical[1], l_horizontal[0], l_horizontal[1]) # top-right
    corner3 = intersect(r_vertical[0], r_vertical[1], r_horizontal[0], r_horizontal[1]) # bottom-right
    corner4 = intersect(l_vertical[0], l_vertical[1], r_horizontal[0], r_horizontal[1]) # bottom-left

    corners = [corner1, corner2, corner3, corner4]
    
    pad = (image_size - board_size) // 2
    # Output square size (you can adjust)
    corners_np = np.float32(corners)  # <-- convert here
    dst = np.float32([[0+pad,0+pad], [board_size+pad,0+pad], [board_size+pad,board_size+pad], [0+pad,board_size+pad]])

    # Perspective transform
    M = cv2.getPerspectiveTransform(corners_np, dst)
    warped = cv2.warpPerspective(frame, M, (image_size, image_size))
    if ref is None:
        #warped = rotate_chessboard(warped, 180)
        return warped
    else:
        warped_aligned, _ = align_to_reference(warped, ref, pad)
        #Change here!
        #warped_aligned = rotate_chessboard(warped_aligned, 180)
        return warped_aligned
def get_border(image, border_thickness=40, pad = 52):
    image = image[pad:-pad, pad:-pad]
    height, width = image.shape[:2]

    # Create a mask with a 100-pixel border
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:border_thickness, :] = 1                # Top border
    mask[-border_thickness:, :] = 1               # Bottom border
    mask[:, :border_thickness] = 1                # Left border
    mask[:, -border_thickness:] = 1               # Right border

    # Apply the mask to each channel
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image
def rotate_chessboard(image, rotation_angle):
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif rotation_angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image
def align_to_reference(image, reference_image, pad = 52):
    """
    Rotate image to match reference orientation.
    Uses template matching or feature detection for robust alignment.
    """
    # Try each rotation and find best match
    best_score = -1
    best_rotation = 0
    
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    ref_cropped = get_border(ref_gray, pad=pad)
    
    for angle in [0, 90, 180, 270]:
        rotated = rotate_chessboard(image, angle)
        rot_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        rot_cropped = get_border(rot_gray, pad = pad)
        
        # Compare using normalized cross-correlation
        result = cv2.matchTemplate(ref_cropped, rot_cropped, cv2.TM_CCOEFF_NORMED)
        score = result[0, 0]
        
        if score > best_score:
            best_score = score
            best_rotation = angle
    
    return rotate_chessboard(image, best_rotation), best_rotation
def get_tiles(image, tile_size=103, num_rows=8, pad = 52):
    num_cols = num_rows
    centers = {}
    for r in range(num_rows):
        for c in range(num_cols):
            x0 = (c * tile_size) + 45 + pad
            y0 = (r * tile_size) + 45 + pad
            x1 = x0 + tile_size
            y1 = y0 + tile_size

            # Draw grid rectangle
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)  # blue border

            # Compute tile center
            cx = x0 + tile_size // 2
            cy = y0 + tile_size // 2
            centers[(r, c)] = (cy, cx)

            cv2.circle(image, (cx, cy), radius=5, color=(0,255,0), thickness=-1)

            # Draw text label
            cv2.putText(image, f"({r},{c})", (cx - 25, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    return image, centers
def get_closest_tile(cx, cy, tile_centers):
    """
    tile_centers: dict {(row, col): (center_x, center_y)}
    (cx, cy): point from contour bounding box center
    """
    min_dist = float("inf")
    best_tile = None

    for tile, (tx, ty) in tile_centers.items():
        dist = (cx - ty)**2 + (cy - tx)**2  # squared distance (faster)
        if dist < min_dist:
            min_dist = dist
            best_tile = tile

    return best_tile