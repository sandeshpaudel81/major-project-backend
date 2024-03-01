import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
from ultralytics import YOLO
from django.apps import apps

def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.032 * peri, True)


# def get_citizenship_contour(contours):
#     # loop over the contours
#     for c in contours:
#         approx = approximate_contour(c)
#         # if our approximated contour has four points, we can assume it is citizenship's rectangle
#         if len(approx) == 4:
#             return approx

def get_citizenship_contour(contours, min_area_threshold=500, min_perimeter_threshold=100):
    citizenship_contour = None
    max_area = 0
    for c in contours:
        approx = approximate_contour(c)
        area = cv2.contourArea(approx)
        peri = cv2.arcLength(approx, True)
        if len(approx) == 4 and area > min_area_threshold and peri > min_perimeter_threshold:
            if area > max_area:
                citizenship_contour = approx
                max_area = area
    return citizenship_contour
        
def contour_to_rect(contour, resize_ratio):
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    # top-left point has the smallest sum
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points:
    # the top-right will have the minumum difference
    # the bottom-left will have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect / resize_ratio

def wrap_perspective(img, rect):
    # unpack rectangle points: top left, top right, bottom right, bottom left
    (tl, tr, br, bl) = rect
    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    # destination points which will be used to map the screen to a "scanned" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    # warp the perspective to grab the screen
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

def get_front_pred(frontImage):
    try:
        nparr = np.fromstring(frontImage.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        resize_ratio = 500 / image.shape[0]
        original = image.copy()
        image = opencv_resize(image, resize_ratio)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     plot_gray(gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Detect white regions
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.dilate(blurred, rectKernel)
        edged = cv2.Canny(dilated, 300, 100, apertureSize=3)
        # Detect all contours in Canny-edged image
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0,255,0), 3)
        # Get 10 largest contours
        largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
        image_with_largest_contours = cv2.drawContours(image.copy(), largest_contours, -1, (0,255,0), 3)
        citizenship_contour = get_citizenship_contour(largest_contours)
        image_with_citizenship_contour = cv2.drawContours(image.copy(), [citizenship_contour], -1, (0, 255, 0), 2)
        #plot_rgb(image_with_citizenship_contour)
        scanned = wrap_perspective(original.copy(), contour_to_rect(citizenship_contour, resize_ratio))
        scanned = cv2.resize(scanned,(2600,1900))
        # plt.figure(figsize=(16,10))
        # plt.imshow(scanned)
        #scanned = cv2.cvtColor(scanned,cv2.COLOR_BGR2RGB)
        # cv2.imwrite("Transformed/"+file_name, scanned)

        # path = BASE_DIR_MODEL / "models/front.pt"
        # model = YOLO(path)
        model = apps.get_app_config('app').front_model
        results = model(scanned)
        r = results[0]
        class_labels = ['office', 'janmasthan', 'permAdd', 'dob', 'father', 'mother', 'spouse', 'number']
        bb = []
        classes = []
        for attr, value in r.__dict__.items():
            if(attr=='boxes'):
                bx = value
                bb = bx.xyxy
                classes = bx.cls
        bounding_boxes = {}
        c = {}
        for i in range(len(classes)):
            class_no = int(classes[i].cpu().numpy())
            class_name = class_labels[class_no]
            bounding_boxes[class_name] = bb[i].cpu().numpy().tolist()
        for i in range(len(class_labels)):
            if class_labels[i] in bounding_boxes:
                c[class_labels[i]+'X'] = int(bounding_boxes[class_labels[i]][0])
                c[class_labels[i]+'Y'] = int(bounding_boxes[class_labels[i]][1])
            else:
                c[class_labels[i]+'X'] = 0
                c[class_labels[i]+'Y'] = 0

        gray_image = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
        average_intensity = np.mean(gray_image)

        if average_intensity <= 165:
            hsv_image = cv2.cvtColor(scanned, cv2.COLOR_BGR2HSV)
            lower=(0,0,0)
            upper=(127,127,99)
            mask=cv2.inRange(scanned,lower,upper)
            result=scanned.copy()
            result[mask!=255]=(255,255,255)

        elif 165< average_intensity <= 182 :
            hsv_image = cv2.cvtColor(scanned, cv2.COLOR_BGR2HSV)

            lower=(0,0,0)
            upper=(160,160,134)
            mask=cv2.inRange(scanned,lower,upper)
            result=scanned.copy()
            result[mask!=255]=(255,255,255)

        else :
            hsv_image = cv2.cvtColor(scanned, cv2.COLOR_BGR2HSV)
            lower=(0,0,0)
            upper=(205,205,180)
            mask=cv2.inRange(scanned,lower,upper)
            result=scanned.copy()
            result[mask!=255]=(255,255,255)

        roi = [
            [
                (max(c['janmasthanX']+459, 0),max(c['janmasthanY']-120, 0)),
                (max(c['janmasthanX']+1204, 0),max(c['janmasthanY'], 0)),
                'text','name'
            ],
            [
                (max(c['dobX']+468, 0),max(c['dobY']-4, 0)),
                (max(c['dobX']+1343, 0),max(c['dobY']+96, 0)),
                'text','dateOfBirth'
            ],
            [
                (max(c['permAddX']+452, 0),max(c['permAddY']-3, 0)),
                (max(c['permAddX']+1151, 0),max(c['permAddY']+84, 0)),
                'text','permanentAddressDistrict'
            ],
            [
                (max(c['janmasthanX']+459, 0),max(c['janmasthanY']-16, 0)),
                (max(c['janmasthanX']+1204, 0),max(c['janmasthanY']+88, 0)),
                'text','placeOfBirthDistrict'
            ],
            [
                (max(c['numberX']+235, 0),max(c['numberY']-44, 0)),
                (max(c['numberX']+833, 0),max(c['numberY']+100, 0)),
                'text','citizenshipNumber'
            ],
            [
                (max(c['officeX']+300, 0),max(c['officeY']-100, 0)),
                (max(c['officeX']+710, 0),max(c['officeY']+120, 0)),
                'text','issuingDistrict'
            ],
            [
                (max(c['janmasthanX']+1228, 0),max(c['janmasthanY']-96, 0)),
                (max(c['janmasthanX']+1900, 0),max(c['janmasthanY']+47, 0)),
                'text','gender'
            ],
            [
                (max(c['janmasthanX']+1228, 0),max(c['janmasthanY']+43, 0)),
                (max(c['janmasthanX']+1860, 0),max(c['janmasthanY']+190, 0)),
                'text','placeOfBirthWard'
            ],
            [
                (max(c['permAddX']+1221, 0),max(c['permAddY']+72, 0)),
                (max(c['permAddX']+1918, 0),max(c['permAddY']+201, 0)),
                'text','permanentAddressWard'
            ],
            [
                (max(c['fatherX']+459, 0),max(c['fatherY']-3, 0)),
                (max(c['fatherX']+1060, 0),max(c['fatherY']+84, 0)),
                'text','fatherName'
            ],
            [
                (max(c['motherX']+475, 0),max(c['motherY']+2, 0)),
                (max(c['motherX']+1026, 0),max(c['motherY']+108, 0)),
                'text','motherName'
            ],
            [
                (max(c['spouseX']+475, 0),max(c['spouseY']+4, 0)),
                (max(c['spouseX']+1026, 0),max(c['spouseY']+114, 0)),
                'text','spouseName'
            ],
            [
                (max(c['permAddX']+452, 0),max(c['permAddY']+80, 0)),
                (max(c['permAddX']+1151, 0),max(c['permAddY']+172, 0)),
                'text','permanentAddressNagarpalika'
            ],
            [
                (max(c['janmasthanX']+459, 0),max(c['janmasthanY']+74, 0)),
                (max(c['janmasthanX']+1204, 0),max(c['janmasthanY']+167, 0)),
                'text','placeOfBirthNagarpalika'
            ]
        ]
        imgshow = result.copy()
        imgMask = np.zeros_like(imgshow)
        front_prediction_result = {}
        for x,r in enumerate(roi):
            cv2.rectangle(imgMask,((r[0][0]),(r[0][1])),((r[1][0]),(r[1][1])),(0,255,0),cv2.FILLED)
            imgshow = cv2.addWeighted(imgshow,0.99,imgMask,0.1,0)
            imgcrop = result[r[0][1]:r[1][1],r[0][0]:r[1][0]]
            output = pytesseract.image_to_string(imgcrop, lang='nep')
            front_prediction_result[r[3]] = output
        return front_prediction_result
    except:
        return {'error': 'Error. Try removing card cover and retake picture in high contrast'}


def get_back_pred(backImage):
    try:
        nparr = np.fromstring(backImage.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        resize_ratio = 500 / image.shape[0]
        original = image.copy()
        image = opencv_resize(image, resize_ratio)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Detect white regions
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.dilate(blurred, rectKernel)
        edged = cv2.Canny(dilated, 300, 100, apertureSize=3)
        # Detect all contours in Canny-edged image
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0,255,0), 3)
        # Get 10 largest contours
        largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
        image_with_largest_contours = cv2.drawContours(image.copy(), largest_contours, -1, (0,255,0), 3)
        citizenship_contour = get_citizenship_contour(largest_contours)
        image_with_citizenship_contour = cv2.drawContours(image.copy(), [citizenship_contour], -1, (0, 255, 0), 2)
        #plot_rgb(image_with_citizenship_contour)
        scanned = wrap_perspective(original.copy(), contour_to_rect(citizenship_contour, resize_ratio))
        scanned = cv2.resize(scanned,(2600,1900))
        # plt.figure(figsize=(16,10))

        # path = BASE_DIR_MODEL / "models/back.pt"
        # model = YOLO(path)
        model = apps.get_app_config('app').back_model
        results = model(scanned)
        r = results[0]
        class_labels = ['type', 'name', 'date']
        bb = []
        classes = []
        for attr, value in r.__dict__.items():
            if(attr=='boxes'):
                bx = value
                bb = bx.xyxy
                classes = bx.cls
        bounding_boxes = {}
        c = {}
        for i in range(len(classes)):
            class_no = int(classes[i].cpu().numpy())
            class_name = class_labels[class_no]
            bounding_boxes[class_name] = bb[i].cpu().numpy().tolist()
        for i in range(len(class_labels)):
            if class_labels[i] in bounding_boxes:
                c[class_labels[i]+'X'] = int(bounding_boxes[class_labels[i]][0])
                c[class_labels[i]+'Y'] = int(bounding_boxes[class_labels[i]][1])
            else:
                c[class_labels[i]+'X'] = 0
                c[class_labels[i]+'Y'] = 0


        gray_image = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
        average_intensity = np.mean(gray_image)
        print(average_intensity)

        if average_intensity <= 165:
            hsv_image = cv2.cvtColor(scanned, cv2.COLOR_BGR2HSV)

            lower=(0,0,0)
            upper=(127,127,99)
            mask=cv2.inRange(scanned,lower,upper)
            result=scanned.copy()
            result[mask!=255]=(255,255,255)

            plt.figure(figsize=(16,10))
            plt.imshow(result)

        elif 165< average_intensity <= 182 :
            hsv_image = cv2.cvtColor(scanned, cv2.COLOR_BGR2HSV)

            lower=(0,0,0)
            upper=(160,160,134)
            mask=cv2.inRange(scanned,lower,upper)
            result=scanned.copy()
            result[mask!=255]=(255,255,255)

            # plt.figure(figsize=(16,10))
            # plt.imshow(result)

        elif 183< average_intensity <= 190 :
            hsv_image = cv2.cvtColor(scanned, cv2.COLOR_BGR2HSV)

            lower=(0,0,0)
            upper=(190,170,154)
            mask=cv2.inRange(scanned,lower,upper)
            result=scanned.copy()
            result[mask!=255]=(255,255,255)

            # plt.figure(figsize=(16,10))
            # plt.imshow(result)

        else :
            hsv_image = cv2.cvtColor(scanned, cv2.COLOR_BGR2HSV)
            lower=(0,0,0)
            upper=(205,205,180)
            mask=cv2.inRange(scanned,lower,upper)
            result=scanned.copy()
            result[mask!=255]=(255,255,255)

        roi = [
            [
            (max(c['typeX']+415, 0),max(c['typeY'], 0)),
            (max(c['typeX']+700, 0),max(c['typeY']+70, 0)),
            'text','type'
            ],
            [
            (max(c['nameX']+190, 0),max(c['nameY'], 0)),
            (max(c['nameX']+690, 0),max(c['nameY']+70, 0)),
            'text','nameOfOfficer'
            ],
            [
            (max(c['dateX']+235, 0),max(c['dateY'], 0)),
            (max(c['dateX']+600, 0),max(c['dateY']+70, 0)),
            'text','dateofissue'
            ],
        ]
        imgshow = result.copy()
        imgMask = np.zeros_like(imgshow)
        back_prediction_result = {}
        for x,r in enumerate(roi):
            cv2.rectangle(imgMask,((r[0][0]),(r[0][1])),((r[1][0]),(r[1][1])),(0,255,0),cv2.FILLED)
            imgshow = cv2.addWeighted(imgshow,0.99,imgMask,0.1,0)
            imgcrop = result[r[0][1]:r[1][1],r[0][0]:r[1][0]]
            output = pytesseract.image_to_string(imgcrop, lang='nep')
            back_prediction_result[r[3]] = output
        return back_prediction_result
    except:
        return {'error': 'Error. Try removing card cover and retake picture in high contrast'}