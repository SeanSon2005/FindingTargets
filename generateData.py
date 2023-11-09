import numpy as np
import cv2
from tqdm.auto import tqdm
import random
import math
import glob
import os
from PIL import Image
import multiprocessing
import colorsys

# folder paths (CHECK BEFORE USING)
BACKGROUND_FOLDER = "C:/Users/Sean/Documents/Coding/YoloCopyLmfao/background_images"
IMAGE_FOLDER = "base_images"
LABELS_FOLDER = "base_labels"

# CHANGEABLE CONSTANTS
N = 100 # total image count will be N * CORE_COUNT!
CORE_COUNT = 20  # the number of logical processors in your system

# Data Augmentation Variables
DATA_AUGMENTATION = {
    "noise_intensity": (0, 30),
    "sun_range": (0.02, 0.05),
    "blur": True
}
FALSE_RATE = 0.2
NUM_SHAPES_MAX = 4
SHAPE_SIZE_RANGE = (25, 35)
FONT = cv2.FONT_HERSHEY_SIMPLEX
IMG_PADDING = 40
CLEAR_FOLDER = True
DRAW_RECT = False

# CONSTANTS
RES = (1280, 720, 4)

# shapes: 0circle, 1semicircle, 2quarter circle, 3triangle, 4rectangle,
# 5pentagon, 6star, and 7cross.
COLORS = {  # the 4th value is for alpha channel
    0: [(0,0,0),(255,15,120)],  # black
    1: [(0,0,230),(255,15,255)],  # white
    2: [(0,200,255),(12,255,255),(345,200,255),(359,255,255)],  # red
    3: [(90,120,255),(160,255,255)],  # green
    4: [(175,120,255),(220,255,255)],  # blue
    5: [(250,120,255),(275,255,255)],  # purple
    6: [(20,120,255),(35,255,255)],  # orange
    7: [(30,120,40),(38,255,60)],  # brown
}

def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

def add_color_variation(color_range):
    if len(color_range) == 4 and random.random() > 0.5:
        h = random.randint(color_range[2][0],color_range[3][0]) / 359
        s = random.randint(color_range[2][1],color_range[3][1]) / 255
        v = random.randint(color_range[2][2],color_range[3][2]) / 255
    else:
        h = random.randint(color_range[0][0],color_range[1][0]) / 359
        s = random.randint(color_range[0][1],color_range[1][1]) / 255
        v = random.randint(color_range[0][2],color_range[1][2]) / 255

    r,g,b = hsv2rgb(h,s,v)
    out = (r,g,b,255)
    return out


# randomly warps the image 


def random_warp(img):
    H,W,_ = img.shape
    original_pts = np.float32([[0,0],[H,0],[0,W],[H,W]])
    warp_style = random.randint(0,3)
    comp = random.random() * 5
    if warp_style == 0:
        warp_plts =  np.float32([[comp,0],[W-comp,0],[0,H],[W,H]])
    elif warp_style == 1:
        warp_plts =  np.float32([[0,0],[W,0],[comp,H],[W-comp,H]])
    elif warp_style == 2:
        warp_plts =  np.float32([[0,comp],[W,0],[0,H-comp],[W,H]])
    else:
        warp_plts =  np.float32([[0,0],[W,comp],[0,H],[W,H-comp]])

    M = cv2.getPerspectiveTransform(original_pts,warp_plts)
    dst = cv2.warpPerspective(img,M,(H,W))

    return dst

# add noise to a single pixel channel


def add_noise(channel_val):
    noise = random.randint(
        DATA_AUGMENTATION["noise_intensity"][0],
        DATA_AUGMENTATION["noise_intensity"][1])
    if channel_val < 128:
        return max(min(channel_val + noise, 255), 0)
    else:
        return max(min(channel_val - noise, 255), 0)

# generate a random character from UPPERCASE ALPHA and NUMBERS


def generate_alphanumeric():
    rng = random.randint(48, 83)
    if rng > 57:
        rng += 7
    return str(chr(rng))

# generate pentagon points from a center point


def pentagon(x, y, a):
    points = []
    for i in range(5):
        angle = i * 72 - 90
        aX = x + a * math.cos(math.radians(angle))
        aY = y + a * math.sin(math.radians(angle))
        points.append((int(aX), int(aY)))
    return np.array(points)

# generate star points from a center point


def star(x, y, a):
    points = []
    for i in range(10):
        angle = i * 36 - 90
        if i % 2 == 0:
            aX = x + a * math.cos(math.radians(angle))
            aY = y + a * math.sin(math.radians(angle))
        else:
            aX = x + a * math.cos(math.radians(angle)) * 0.422
            aY = y + a * math.sin(math.radians(angle)) * 0.422
        points.append((int(aX), int(aY)))
    return np.array(points)

# generate cross poionts from a center point


def cross(x, y, a):
    points_a = [(x - a, y + int(a * (0.5))), (x + a, y - int(a * (0.5)))]
    points_b = [(x - int(a * (0.5)), y + a), (x + int(a * (0.5)), y - a)]
    return points_a, points_b


def getLabel(x, y, a):
    x_center = str(x / RES[0])
    y_center = str(y / RES[1])
    x_end = str(a / RES[0])
    y_end = str(a / RES[1])
    return x_center + " " + y_center + " " + x_end + " " + y_end + "\n"


def image_process(color1, color2):
    print(color1)
    if color1[3] == 0:
        color = color2
    else:
        color = color1
        color[3] == 255
    return color

# generate image function


def generate_image(iter, consts):
    img_name = "Image" + str(iter) + ".jpg"

    num_shapes = random.randint(1, NUM_SHAPES_MAX)
    if NUM_SHAPES_MAX == 0 or FALSE_RATE == 0 or (
            random.randint(0, int(1 / FALSE_RATE)) == 0):
        num_shapes = 0

    # load background image
    bg_image_id = random.randint(1, 500)
    bg_img = cv2.imread(
        BACKGROUND_FOLDER +
        "/bg_img" +
        str(bg_image_id) +
        ".jpg")
    #bg_img = bg_img[:,:RES[0]]

    with open(LABELS_FOLDER + '/Image' + str(iter) + '.txt', 'w') as f:
        for i in range(num_shapes):
            shape_id = random.randint(0, 7)

            # handle sizes
            text_size, text_thickness = 1.5, 3

            img_size = 100 + IMG_PADDING
            img = np.zeros((img_size, img_size, 4)).astype(np.uint8)
            x = 50 + int(IMG_PADDING / 2)
            y = 50 + int(IMG_PADDING / 2)

            img_out_size = random.randint(
                SHAPE_SIZE_RANGE[0], SHAPE_SIZE_RANGE[1])

            alphanumeric = generate_alphanumeric()
            rotation = random.randint(0, 359)

            # handle color decisions
            color_num_1 = random.randint(0, 7)
            color_num_2 = random.randint(0, 7)
            while (color_num_1 == color_num_2):
                color_num_2 = random.randint(0, 7)
            color = add_color_variation(COLORS[color_num_1])
            colorText = add_color_variation(COLORS[color_num_2])

            textsize = cv2.getTextSize(alphanumeric, FONT, text_size, 2)[0]
            text_point = (x - int(textsize[0] / 2), y + int(textsize[1] / 2))

            if shape_id == 0:  # circle
                cv2.circle(img, (x, y), 50, color, -1)
                cv2.putText(img,
                            alphanumeric,
                            text_point,
                            FONT,
                            text_size,
                            colorText,
                            text_thickness)

            if shape_id == 1:  # half circle
                cv2.ellipse(img, (x, y + 27), (55,
                            55), 0, 0, -180, color, -1)
                cv2.putText(img,
                            alphanumeric,
                            text_point,
                            FONT,
                            text_size,
                            colorText,
                            text_thickness)

            if shape_id == 2:  # quarter circle
                cv2.ellipse(img, (x - 33, y - 33), (80,
                            80), 0, 0, 90, color, -1)
                cv2.putText(img,
                            alphanumeric,
                            text_point,
                            FONT,
                            text_size,
                            colorText,
                            text_thickness)

            if shape_id == 3:  # triangle
                cv2.drawContours(img,
                                 [np.array([(x - 50,
                                            y + int(25 *
                                                    consts["SQRT3"]) - 15),
                                           (x + 50,
                                            y + int(25 *
                                                    consts["SQRT3"]) - 15),
                                            (x,
                                            y - int(25 *
                                                    consts["SQRT3"]) - 15)])],
                                 0,
                                 color,
                                 -1)
                cv2.putText(img,
                            alphanumeric,
                            text_point,
                            FONT,
                            text_size,
                            colorText,
                            text_thickness)

            if shape_id == 4:  # rectangle
                cv2.rectangle(img,
                              (x - 50,
                               y - 50),
                              (x + 50,
                               y + 50),
                              color, -1)
                cv2.putText(img,
                            alphanumeric,
                            text_point,
                            FONT,
                            text_size,
                            colorText,
                            text_thickness)

            if shape_id == 5:  # pentagon
                cv2.drawContours(
                    img, [
                        pentagon(x, y, 50)], 0, color, -1)
                cv2.putText(img,
                            alphanumeric,
                            text_point,
                            FONT,
                            text_size,
                            colorText,
                            text_thickness)

            if shape_id == 6:  # star
                cv2.drawContours(
                    img, [
                        star(x, y, 60)], 0, color, -1)
                cv2.putText(img,
                            alphanumeric,
                            text_point,
                            FONT,
                            text_size,
                            colorText,
                            text_thickness)

            if shape_id == 7:  # cross
                points_a, points_b = cross(x, y, 50)
                cv2.rectangle(img, points_a[0], points_a[1], color, -1)
                cv2.rectangle(img, points_b[0], points_b[1], color, -1)
                cv2.putText(img,
                            alphanumeric,
                            text_point,
                            FONT,
                            text_size,
                            colorText,
                            text_thickness)

            frame_x = random.randint(0, RES[0] - img_out_size)
            frame_y = random.randint(0, RES[1] - img_out_size)

            # resize image
            img = cv2.resize(img, (img_out_size, img_out_size))

            # rotate image
            M = cv2.getRotationMatrix2D(
                (int(img_out_size / 2), int(img_out_size / 2)), rotation, 1)
            img = cv2.warpAffine(img, M, (img_out_size, img_out_size))

            for ay in range(img_out_size):
                for ax in range(img_out_size):
                    y_i = frame_y + ay
                    x_i = frame_x + ax
                    if y_i >= RES[1] or x_i >= RES[0] or y_i < 0 or x_i < 0:
                        continue
                    if img[ay][ax][3] == 255:
                        bg_img[y_i][x_i][0] = add_noise(img[ay][ax][0])
                        bg_img[y_i][x_i][1] = add_noise(img[ay][ax][1])
                        bg_img[y_i][x_i][2] = add_noise(img[ay][ax][2])

            text_pixels = int(img_out_size / 2.3)
            if DRAW_RECT:
                topLeftX = frame_x + int(img_out_size / 2) - text_pixels
                topLeftY = frame_y + int(img_out_size / 2) - text_pixels
                cv2.rectangle(bg_img,(topLeftX,topLeftY),(frame_x +
                        int(img_out_size / 2) + text_pixels,frame_y +
                        int(img_out_size / 2) + text_pixels),(0,0,0),2)
            f.write(
                "0 " +
                getLabel(
                    frame_x + int(img_out_size / 2),
                    frame_y + int(img_out_size / 2),
                    text_pixels*2))

    # Add sun filter (adds a warm filter)
    sun_filter = np.uint8(np.ones_like(bg_img) * (255, 180, 0))
    amount = ((random.random() * (DATA_AUGMENTATION["sun_range"][1] -
                                  DATA_AUGMENTATION["sun_range"][0])) +
              DATA_AUGMENTATION["sun_range"][0])
    bg_img = cv2.addWeighted(sun_filter, amount, bg_img, 1 - amount, 0)

    # Add camera blur
    if DATA_AUGMENTATION["blur"]:
        if  random.randint(0, 1) == 1:
            bg_img = cv2.GaussianBlur(bg_img, (3, 3), 0)

    # Saves image as jpg
    bg_img = Image.fromarray(bg_img)
    bg_img.save(IMAGE_FOLDER + "/" + img_name)


# Actual Image Generation
if __name__ == '__main__':
    # calculation constants (so we don't have to recalculate and just pass
    # through memory)
    EMPTY_SHAPE_IMAGE = np.zeros(
        (RES[0], RES[1], 4), dtype=np.uint8)
    SQRT3 = math.sqrt(3)
    CONSTANTS = {
        "SQRT3":SQRT3,
        "EMPTY_SHAPE_IMAGE": EMPTY_SHAPE_IMAGE
    }

    # clear generated files
    if CLEAR_FOLDER:
        files = glob.glob(IMAGE_FOLDER + "/*")
        for f in files:
            os.remove(f)
        files = glob.glob(LABELS_FOLDER + "/*")
        for f in files:
            os.remove(f)
    try:
        pool = multiprocessing.Pool(processes=CORE_COUNT)

        # Process Images Standard.
        for i in tqdm(range(N), desc="generating data"):
            # Multi Process Images
            index = i * CORE_COUNT
            iter = np.zeros(CORE_COUNT, dtype=np.object_)
            for i in range(CORE_COUNT):
                iter[i] = (index + i, CONSTANTS)
            pool.starmap(generate_image, iter)

    finally:
        pool.close()
        pool.join()
