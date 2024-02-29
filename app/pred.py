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


def get_citizenship_contour(contours):
    # loop over the contours
    for c in contours:
        approx = approximate_contour(c)
        # if our approximated contour has four points, we can assume it is citizenship's rectangle
        if len(approx) == 4:
            return approx
        
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
    print(average_intensity)

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
        [(c['janmasthanX']+459,c['janmasthanY']-120),(c['janmasthanX']+1204,c['janmasthanY']),'text','name'],
        [(c['dobX']+468,c['dobY']-4),(c['dobX']+1343,c['dobY']+96),'text','dateOfBirth'],
        [(c['permAddX']+452,c['permAddY']-3),(c['permAddX']+1151,c['permAddY']+84),'text','permanentAddressDistrict'],
        [(c['janmasthanX']+459,c['janmasthanY']-16),(c['janmasthanX']+1204,c['janmasthanY']+88),'text','placeOfBirthDistrict'],
       [(c['numberX']+235,c['numberY']-44),(c['numberX']+833,c['numberY']+100),'text','citizenshipNumber'],
        [(c['officeX']+300,c['officeY']-100),(c['officeX']+710,c['officeY']+120),'text','issuingDistrict'],
        [(c['janmasthanX']+1228,c['janmasthanY']-96),(c['janmasthanX']+1900,c['janmasthanY']+47),'text','gender'],
        [(c['janmasthanX']+1228,c['janmasthanY']+43),(c['janmasthanX']+1860,c['janmasthanY']+190),'text','placeOfBirthWard'],
        [(c['permAddX']+1221,c['permAddY']+72),(c['permAddX']+1918,c['permAddY']+201),'text','permanentAddressWard'],
        [(c['fatherX']+459,c['fatherY']-3),(c['fatherX']+1060,c['fatherY']+84),'text','fatherName'],
        [(c['motherX']+475,c['motherY']+2),(c['motherX']+1026,c['motherY']+108),'text','motherName'],
        [(c['spouseX']+475,c['spouseY']+4),(c['spouseX']+1026,c['spouseY']+114),'text','spouseName'],
       [(c['permAddX']+452,c['permAddY']+80),(c['permAddX']+1151,c['permAddY']+172),'text','permanentAddressNagarpalika'],
        [(c['janmasthanX']+459,c['janmasthanY']+74),(c['janmasthanX']+1204,c['janmasthanY']+167),'text','placeOfBirthNagarpalika']
      ]
    imgshow = result.copy()
    imgMask = np.zeros_like(imgshow)
    front_prediction_result = {}
    for x,r in enumerate(roi):
        cv2.rectangle(imgMask,((r[0][0]),(r[0][1])),((r[1][0]),(r[1][1])),(0,255,0),cv2.FILLED)
        imgshow = cv2.addWeighted(imgshow,0.99,imgMask,0.1,0)
        # print(result.shape)
        # print(r[0][1])
        # print(r[1][1])
        # print(r[0][0])
        # print(r[1][0])
        imgcrop = result[r[0][1]:r[1][1],r[0][0]:r[1][0]]
        output = pytesseract.image_to_string(imgcrop, lang='nep')
        front_prediction_result[r[3]] = output
    return front_prediction_result


def get_back_pred(backImage):
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
        [(c['typeX']+415,c['typeY']),(c['typeX']+700,c['typeY']+70),'text','type'],
        [(c['nameX']+190,c['nameY']),(c['nameX']+690,c['nameY']+70),'text','nameOfOfficer'],
        [(c['dateX']+235,c['dateY']),(c['dateX']+600,c['dateY']+70),'text','dateofissue'],
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