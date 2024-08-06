import cv2
import numpy as np

def  Douglas_Peucker(mask_path, output_path, epsilon=0.01):
    # Load the mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Apply closing operation to smooth the boundaries
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image to draw the polygons
    polygonal_mask = np.zeros_like(mask)

    for contour in contours:
        # Approximate contour to a polygon
        epsilon_value = epsilon * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon_value, True)
        cv2.drawContours(polygonal_mask, [polygon], -1, (255), thickness=cv2.FILLED)

    # Save the resulting mask
    cv2.imwrite(output_path, polygonal_mask)

# Usage example
# Douglas_Peucker('capture/test/area_0.png', 'capture/test/area_0_aa.png', epsilon=0.003)
