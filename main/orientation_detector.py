from doctest import debug
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import cv2
import os
import numpy as np
import sys
from sklearn import linear_model



def detect_orientation(warped_wide_board, use_inv=False, bw_threshold=110, debug=False):
    rotated_images = get_rotated_images(warped_wide_board)
    width, height = warped_wide_board.size
    best_detection_edge_rotation, best_detection_edge_detections_rotated, best_detection_edge_detections_original = find_best_detection_edge(rotated_images, width, height, use_inv=use_inv, bw_threshold=bw_threshold, debug=debug)
    if best_detection_edge_rotation is None:
        return None
    white_edge = find_white_edge(best_detection_edge_rotation, best_detection_edge_detections_rotated, debug=debug)

    if debug:
        warped_wide_board_cv2 = cv2.cvtColor(np.array(warped_wide_board), cv2.COLOR_RGB2BGR)
        if best_detection_edge_detections_original is not None:
            for detection in best_detection_edge_detections_original:
                letter, x_center, y_center, size = detection
                x1 = int(x_center - size / 2)
                y1 = int(y_center - size / 2)
                x2 = int(x_center + size / 2)
                y2 = int(y_center + size / 2)

                cv2.rectangle(warped_wide_board_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(warped_wide_board_cv2, letter, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Detected Letters on Board", warped_wide_board_cv2)
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)

    return white_edge

def find_white_edge(detection_edge_rotation, detection_edge_detections_rotated, debug=False):
    ransac_result = ransac_line_fitting(detection_edge_detections_rotated)
    if ransac_result is None:
        return None
    line_params, inlier_mask = ransac_result
    if line_params is None:
        return None
    slope, intercept = line_params
    if abs(slope) > 0.5:
        if debug:
            print("Detected line slope too steep.")
        return None
    
    correct_detections = []
    for idx, is_inlier in enumerate(inlier_mask):
        if is_inlier:
            correct_detections.append(detection_edge_detections_rotated[idx])

    if len(correct_detections) < 2:
        if debug:
            print("Not enough inlier detections for reliable edge.")
        return None

    letters = ["a","b","c","d","e","f","g","h"]
    reverse_count = 0
    forward_count = 0
    for i in range(len(correct_detections)-1):
        for j in range(i+1, len(correct_detections)):
            d1 = correct_detections[i]
            d2 = correct_detections[j]
            letter1, x1, y1, size1 = d1
            letter2, x2, y2, size2 = d2
            if letter1 == letter2:
                continue
            letter1_idx = letters.index(letter1)
            letter2_idx = letters.index(letter2)
            if x1 < x2 and letter1_idx < letter2_idx:
                forward_count += 1
            elif x1 > x2 and letter1_idx > letter2_idx:
                forward_count += 1
            else:
                reverse_count += 1
    if debug:
        print(f"Edge {detection_edge_rotation}: forward count = {forward_count}, reverse count = {reverse_count}")
    print(f"Edge {detection_edge_rotation}: forward count = {forward_count}, reverse count = {reverse_count}")
    if forward_count == reverse_count:
        if debug:
            print("Unable to determine white edge orientation due to equal forward and reverse counts.")
        return None
    if forward_count > reverse_count:
        return detection_edge_rotation
    else:
        return (detection_edge_rotation + 2) % 4

def ransac_line_fitting(detections):
    if len(detections) < 2:
        return None, None
    points = np.array([[d[1], d[2]] for d in detections])  
    X, y = points[:, 0].reshape(-1, 1), points[:, 1]
    ransac = linear_model.RANSACRegressor(
        estimator=linear_model.LinearRegression(),
        residual_threshold=15,
        min_samples=3,
        max_trials=1000,
        random_state=0
    )
    ransac.fit(X, y)

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    line_coef = ransac.estimator_.coef_[0]
    line_intercept = ransac.estimator_.intercept_

    return (line_coef, line_intercept), inlier_mask

def find_best_detection_edge(images, width, height, use_inv=False, bw_threshold=110, debug=False):
    detections_per_edge = []
    for idx, image in enumerate(images):
        crop_height_scale = 0.8
        crop_box = (0, height*crop_height_scale, width, height)
        region, img_to_ocr, data = ocr_region_only_letters(image, use_inv=use_inv, bw_threshold=bw_threshold, upscale_factor=4, box=crop_box)

        detected_letters = []
        for index, row in data.iterrows():
            x = row['left']
            y = row['top']
            w = row['width']
            h = row['height']
            conf = row['conf']
            letter = row['text']

            if w > 25 or h > 25 or conf < 1 or letter not in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
                continue

            x_center = x + w / 2
            y_center = y + h / 2 + height * crop_height_scale
            size = max(w, h)

            detected_letters.append([letter, x_center, y_center, size])

        if debug:
            print(f"Detected {len(detected_letters)} letters on edge {idx}.")
            print(data)
            if idx == 0:
                debug_image1 = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)
                debug_image2 = cv2.cvtColor(np.array(img_to_ocr), cv2.COLOR_RGB2BGR)
                cv2.imshow(f"Region", debug_image1)
                cv2.imshow(f"Image to OCR", debug_image2)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    sys.exit(0)

        detections_per_edge.append(detected_letters)

    best_edge_index = np.argmax([len(d) for d in detections_per_edge])
    best_edge_detections_rotated = detections_per_edge[best_edge_index]

    if debug:
        print(f"Best edge is {best_edge_index} with {len(best_edge_detections_rotated)} detections.")
        best_detection_img = images[best_edge_index]
        cv2_best_detection_img = cv2.cvtColor(np.array(best_detection_img), cv2.COLOR_RGB2BGR)
        for detection in best_edge_detections_rotated:
            letter, x_center, y_center, size = detection
            x1 = int(x_center - size / 2)
            y1 = int(y_center - size / 2)
            x2 = int(x_center + size / 2)
            y2 = int(y_center + size / 2)

            cv2.rectangle(cv2_best_detection_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(cv2_best_detection_img, letter, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.imshow("Best Edge Detections", cv2_best_detection_img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)

    if len(best_edge_detections_rotated) >= 3:
        best_edge_detections_original = []
        for detection in best_edge_detections_rotated:
            letter, x_center, y_center, size = detection
            x_center_original, y_center_original = rotate_point_based_on_rotation_idx(x_center, y_center, best_edge_index, width, height)
            best_edge_detections_original.append([letter, x_center_original, y_center_original, size])
        return best_edge_index, best_edge_detections_rotated, best_edge_detections_original
    return None, None, None

def get_rotated_images(image):
    images = []
    img = image.copy()
    images.append(img)
    for i in range(3):
        cpy_img = image.copy()
        cpy_img = cpy_img.rotate((i+1)*90, expand=True)
        images.append(cpy_img)
    return images

def rotate_point_based_on_rotation_idx(x, y, rotation_idx, width, height):
    if rotation_idx == 0:
        return x, y
    elif rotation_idx == 1:
        return height - y, x
    elif rotation_idx == 2:
        return width - x, height - y
    elif rotation_idx == 3:
        return y, width - x

def ocr_region_only_letters(img, psm=6, whitelist="abcdefgh", use_inv=False, bw_threshold=110, upscale_factor=1, box=None):
    region = img.crop(box) if box else img
    return_region = region.copy()

    if upscale_factor != 1:
        region = region.resize((region.width*upscale_factor, region.height*upscale_factor), Image.Resampling.LANCZOS)

    gray = ImageOps.grayscale(region)
    gray = ImageEnhance.Contrast(gray).enhance(3.5)

    img_to_ocr = gray
    if use_inv:
        bw = gray.point(lambda x: 255 if x > bw_threshold else 0)
        inv = ImageOps.invert(bw)
        img_to_ocr = inv

    config = (
        f"--oem 1 --psm {psm} "
        f"-c tessedit_char_whitelist={whitelist} "
        f"-c load_system_dawg=0 -c load_freq_dawg=0 "
        f"-c user_defined_dpi=300"
    )

    
    data = pytesseract.image_to_data(img_to_ocr, output_type=pytesseract.Output.DATAFRAME, config=config)
    data = data.dropna(subset=['text'])
    data['text'] = data['text'].astype(str)
    data['text'] = data['text'].str.strip()
    data = data[data['text']!=""]

    if upscale_factor != 1:
        data['left'] = (data['left'] / upscale_factor).astype(int)
        data['top'] = (data['top'] / upscale_factor).astype(int)
        data['width'] = (data['width'] / upscale_factor).astype(int)
        data['height'] = (data['height'] / upscale_factor).astype(int)
    else:
        data['left'] = (data['left']).astype(int)
        data['top'] = (data['top']).astype(int)
        data['width'] = (data['width']).astype(int)
        data['height'] = (data['height']).astype(int)
    return return_region, img_to_ocr, data


if __name__ == "__main__":
    image_path = "warped_wide_boards/warped_wide_board.jpg"
    # image_path = "warped_wide_boards/warped_wide_board_20251219_112329.jpg"
    # image_path = "warped_wide_boards/warped_wide_board_20251219_112139.jpg"
    warped_wide_board = Image.open(image_path)
    detect_orientation(warped_wide_board, debug=True)