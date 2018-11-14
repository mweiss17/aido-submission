
import numpy as np
from PIL import Image
import cv2
import math
from math import hypot
from PIL import Image, ImageDraw

RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img

def average_slope_intercept(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # add more weight to longer lines
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None

    return left_lane, right_lane # (slope, intercept), (slope, intercept)

def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)

    y1 = image.shape[0] # bottom of the image
    y2 = y1*0.3         # slightly lower than the middle

    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    return left_line, right_line


def draw_lane_lines(image, lines, thickness=2):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    colors = [[255, 0, 0], [0, 255, 0]]
    for idx, line in enumerate(lines):
        cv2.line(line_image, line[0], line[1],  colors[idx], thickness)
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)



def calc_distance_lane(points):
    x1, y1, x2, y2 = deconstruct_lane_lines(points)
    return math.hypot(x2-x1, y2-y1)

def calc_distance(x1, y1, x2, y2):
    return math.hypot(x2-x1, y2-y1)

def deconstruct_lane_lines(points):
    p1, p2 = points
    x1, y1 = p1
    x2, y2 = p2
    return x1, y1, x2, y2

def slope(points):
    x1, y1, x2, y2 = deconstruct_lane_lines(points)
    return -((y2-y1)/(x2-x1))

def angle_to_action(angle):
    vel = .8
    gain = 1
    trim = 0
    radius = 0.318
    k = 27.0
    wheel_dist =.102
    limit = 1.0

    # assuming same motor constants k for both motors
    k_r = 1
    k_l = 1

    # adjusting k by gain and trim
    k_r_inv = (gain + trim) / k_r
    k_l_inv = (gain - trim) / k_l

    omega_r = (vel + 0.5 * angle * wheel_dist) / radius
    omega_l = (vel - 0.5 * angle * wheel_dist) / radius

    # conversion from motor rotation rate to duty cycle
    u_r = omega_r * k_r_inv
    u_l = omega_l * k_l_inv

    # limiting output to limit, which is 1.0 for the duckiebot
    u_r_limited = max(min(u_r, limit), .3)
    u_l_limited = max(min(u_l, limit), .3)
    action = np.array([u_l_limited, u_r_limited])
    return action

class model:
    def predict(self, observation, trial=0, step=0):
        observation = (np.transpose(observation, (1, 2, 0)) * 255).astype('uint8')
        i = Image.fromarray(observation, 'RGB')
        #plt.figure()
        #plt.imshow(i)

        # Canny-ify
        image = cv2.dilate(observation, None)
        image = cv2.erode(image, None)
        # plt.figure()
        # plt.imshow(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        r, thresh_image = cv2.threshold(gray_image, 127, 255, 1)
        cannyed_image = cv2.Canny(thresh_image, 100, 200)
        # plt.figure()
        # plt.imshow(cannyed_image)

        # Hough Lines

        lines = cv2.HoughLinesP(cannyed_image, rho=1, theta=np.pi/180, threshold=10, minLineLength=20, maxLineGap=10)
        if not lines:
            return [1, 1]
        #print(len(lines))
        line_img = draw_lines(image, lines)
        #plt.figure()
        #plt.imshow(line_img)

        # Lane Lines
        print(lines)
        lane_lines_image = Image.fromarray(draw_lane_lines(image, lane_lines(image, lines)), 'RGB')
        # lane_lines_image.save("/tmp/some_img.jpg")
        #plt.figure()
        #plt.imshow(lane_lines_image)

        # Slope Calulations
        left_lane_line = lane_lines(image, lines)[0]
        right_lane_line = lane_lines(image, lines)[1]
        #left_length = calc_distance_lane(left_lane_line)
        #right_length = calc_distance_lane(right_lane_line)
        try:
            left_slope = -math.atan(slope(left_lane_line)) *(180 / math.pi)
            right_slope = -math.atan(slope(right_lane_line)) *(180 / math.pi)
        except Exception:
            return [1., 1.]
        slope_to_use = right_slope
        using_slope = "right"
        if abs(left_slope) > abs(right_slope):
            slope_to_use = left_slope
            using_slope = "left"

        print("angle: " + str(slope_to_use))
        action = angle_to_action(slope_to_use)


        d = ImageDraw.Draw(lane_lines_image)
        d.text((10,10), "step: " + str(step), fill=(255,255,0))
        d.text((10,20), "action: " + str(action), fill=(255,255,0))
        d.text((10,40), "left_slope: " + str(left_slope), fill=(255,255,0))
        d.text((10,50), "right_slope: " + str(right_slope), fill=(255,255,0))
        d.text((10,60), "using_slope: " + str(using_slope), fill=(255,255,0))
        lane_lines_image.save("/tmp/results/" + str(trial) + "/" + str(step) + "_lane_lines.jpg")

        return action

    def close(self):
        pass
