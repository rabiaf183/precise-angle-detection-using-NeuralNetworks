import cv2
import numpy as np
import math
import random
import csv
import os

# Define the file to save images
image_folder = './generated_images'
csv_filename = './rectangle_angles.csv'

# Create the image folder if it doesn't exist
os.makedirs(image_folder, exist_ok=True)

# Empty List to store angle data
angles = []

# Define the fixed grayscale level for the rectangle
fixed_rectangle_gray_level = 150

# Generate 1200 images
for i in range(1200):
    # Image and rectangle properties
    width, height = 600, 400
    rectangle_width, rectangle_height = int(0.6 * width), int(0.6 * height)
    angle_degrees = random.uniform(-10, 10)

    # Calculate bounding radius to keep rectangle within image bounds
    bounding_radius = math.sqrt((rectangle_width / 2)**2 + (rectangle_height / 2)**2)

    # Safe area for the rectangle center
    safe_x_margin, safe_y_margin = bounding_radius, bounding_radius
    center_x = random.uniform(safe_x_margin, width - safe_x_margin)
    center_y = random.uniform(safe_y_margin, height - safe_y_margin)

    # Create a blank grayscale image
    image = np.full((height, width), 128, dtype=np.uint8)  # 128 for mid-gray background

    # Calculate rectangle corner points
    rect_points = np.array([
        [center_x - rectangle_width / 2, center_y - rectangle_height / 2],
        [center_x + rectangle_width / 2, center_y - rectangle_height / 2],
        [center_x + rectangle_width / 2, center_y + rectangle_height / 2],
        [center_x - rectangle_width / 2, center_y + rectangle_height / 2]
    ], dtype=np.float32)

    # Apply rotation
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_degrees, 1.0)
    rotated_points = cv2.transform(rect_points.reshape(-1, 1, 2), rotation_matrix)

    # Draw the rectangle
    cv2.fillPoly(image, [np.int32(rotated_points)], fixed_rectangle_gray_level)

    # Save the image
    image_filename = os.path.join(image_folder, f'rotated_rectangle_{i}.png')
    cv2.imwrite(image_filename, image)

    # Record the angle
    angles.append((f'rotated_rectangle_{i}', angle_degrees))

# Save angles to a CSV file
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['filename', 'angle'])
    csv_writer.writerows(angles)

print(f'Angles saved to {csv_filename}')
print(f'Images saved to {image_folder}')
