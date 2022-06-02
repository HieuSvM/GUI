import dlib
import torch
import numpy as np
from torchvision import transforms
import os
import sys
import time
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import cv2
from tkinter import filedialog
from PIL import Image
import time
import PIL.Image, PIL.ImageTk
from tkinter import *
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
# load model
def loadmd():
    global modelfull,modelmask,model,transforms_test,modelf,device
    modelfull = load_model('model.h5')
    modelmask = load_model('modelmask.h5')
    model = load_model("mask_detector.model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelf = torch.load('E:/Desktop/samsung/Doan/extract512/InceptionResNetV1_ArcFace.pt')
    modelf = modelf.to(device)
    transforms_list2 = [transforms.Resize((128, 128)), transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    transforms_test = transforms.Compose(transforms_list2)
    modelf.eval()
loadmd()
namemask = []
label1 = " "
namefull = []
root = Tk()
root.title("NHẬN DẠNG KHUÔN MẶT")
root.configure(background='white')
root.resizable(False, False)
window_height = 700
window_width = 1000

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x_cordinate = int((screen_width / 2) - (window_width / 2))
y_cordinate = int((screen_height / 2) - (window_height / 2))

root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
def data():
    for f in os.listdir('./training_dataset'):
        namefull.append(f)
    names_encode = LabelEncoder().fit(namefull)
    names_encode.transform(namefull).tolist()  # [0,1,2,3,4,5,6,7]
    for f in os.listdir('./training_mask'):
        namemask.append(f)
    names_encode1 = LabelEncoder().fit(namemask)
    names_encode1.transform(namemask).tolist()  # [0,1,2,3,4,5,6,7]
data()
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)
frame1 = LabelFrame(root,background="Light Blue")
frame2 = Label(frame1,background="Light Blue")
frame3 = LabelFrame(root,background="")

check = True
def npf32u8(np_arr):
    # intensity conversion
    if str(np_arr.dtype) != 'uint8':
        np_arr = np_arr.astype(np.float32)
        np_arr -= np.min(np_arr)
        np_arr /= np.max(np_arr)  # normalize the data to 0 - 1
        np_arr = 255 * np_arr  # Now scale by 255
        np_arr = np_arr.astype(np.uint8)
    return np_arr


def opencv2pil(opencv_image):
    opencv_image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
    opencv_image_rgb = npf32u8(opencv_image_rgb)  # convert numpy array type float32 to uint8
    pil_image = PIL.Image.fromarray(opencv_image_rgb)  # convert numpy array to Pillow Image Object
    return pil_image
logo = cv2.imread("Logo.jpg")
logo = cv2.resize(logo,(295,700), interpolation = cv2.INTER_AREA)
logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
logo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(np.asarray(logo)))
Label(root,image=logo).pack(side = RIGHT)
my_listbox = Listbox(frame3,height=0,font=("Arial Bold", 9))
my_listbox.pack(side=LEFT,fill="both", expand="yes")
my_listbox.insert(END, "Image")
my_listbox.insert(END, "Video")
my_listbox.insert(END, "Camera Live")
def openfile():
    global file, check,vv, nguong
    check = False
    vv = my_listbox.get(ANCHOR)
    if (my_listbox.get(ANCHOR)== "Image"):
        file = filedialog.askopenfilename(title='open')
        check = True
        file = cv2.imread(file)
        nguong = 0.98
        checkimg(file)
    elif (my_listbox.get(ANCHOR)== "Video"):
        file = filedialog.askopenfilename(title='open')
        vs = cv2.VideoCapture(file)
        check = True
        nguong = 0.98
        videoo(vs)
    elif (my_listbox.get(ANCHOR)== "Camera Live"):
        file = 0
        vs1 = cv2.VideoCapture(file)
        nguong = 0.7
        check = True
        checkcam(vs1)
def detect_and_predict_mask(frame, faceNet, maskNet,dis):
    try:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()
        faces = []
        locs = []
        preds = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > dis:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                face = frame[startY:endY, startX:endX]
                if face.any():
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))
        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)
        return (locs, preds)
    except:
        pass
cop = " "
RGB = None
isprocess = False
def checkk(img):
    global label1,cop,elapsed_time,RGB,isprocess,label
    isprocess = True
    try:
        start_time = time.time()
        if label == "No Mask":
            tmp1 = img[startY:endY, startX:endX]
            tmp1 = cv2.cvtColor(tmp1, cv2.COLOR_BGR2RGB)
            img1_detection = detector(tmp1, 1)
            if len(img1_detection) == 1:
                img1_shape = sp(tmp1, img1_detection[0])
                img1_aligned = dlib.get_face_chip(tmp1, img1_shape)
                tmp1 = facerec.compute_face_descriptor(img1_aligned)
                tmp1 = np.array(tmp1)
                tmp1 = tmp1.reshape(1, 128)
                pre_full = modelfull.predict(tmp1)
                ind = np.argsort(pre_full[0])
                for b in range(len(pre_full[0])):
                    if pre_full[0][b] == pre_full[0][ind[-1]]:
                        if ((pre_full[0][ind[-1]]) * 100 > 95):
                            label1 = "{}: {:.2f}% ".format(namefull[b], (pre_full[0][ind[-1]]) * 100) + label
                        else:
                            label1 = "Unknown " + label
        else:
            with torch.no_grad():
                tmp2 = img[startY:endY, startX:endX]
                tmp2 = cv2.cvtColor(tmp2, cv2.COLOR_BGR2RGB)
                tmp2 = opencv2pil(tmp2)
                tmp2 = transforms_test(tmp2)
                tmp2 = tmp2.unsqueeze(0)
                tmp2 = tmp2.to(device)
                tmp2 = modelf(tmp2)
                tmp2 = tmp2['embeddings'].cpu().detach().numpy()
                pre_mask = modelmask.predict(tmp2)
                ind = np.argsort(pre_mask[0])
                for b in range(len(pre_mask[0])):
                    if pre_mask[0][b] == pre_mask[0][ind[-1]]:
                        if ((pre_mask[0][ind[-1]]) * 100 > 95):
                            label1 = "{}: {:.2f}% ".format(namemask[b], (pre_mask[0][ind[-1]]) * 100) + label
                        else:
                            label1 = "Unknown " + label
        end_time = time.time()
        elapsed_time = end_time - start_time
        #print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    except:
        pass
    isprocess = False
import threading
dem=0
color = (0, 0, 255)
def checkimg(frame):
    global label, startX, startY, endX, endY, label1, color
    if (frame.shape[1] > 550) & (frame.shape[1] > frame.shape[0]):
        frame = imutils.resize(frame, width=550)
    elif (frame.shape[0] > 550) & (frame.shape[0] > frame.shape[1]):
        frame = imutils.resize(frame, height=550)
    (locs, preds) = detect_and_predict_mask(frame, net, model,nguong)
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        checkk(frame)
        cv2.putText(frame, label1, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
    RGB = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(np.asarray(RGB)))
    frame2.configure(image=RGB)
    frame2.image = RGB
def videoo(qt):
    global isprocess,label,startX, startY, endX, endY,cop,dem,label1,color
    try:
        _, frame = qt.read()
        label1 = cop
        start_time = time.time()
        if (frame.shape[1] > 550) & (frame.shape[1] > frame.shape[0]):
            frame = imutils.resize(frame, width=550)
        elif (frame.shape[0] > 550) & (frame.shape[0] > frame.shape[1]):
            frame = imutils.resize(frame, height=550)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        if dem>=3:
            (locs, preds) = detect_and_predict_mask(frame, net, model,nguong)
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                if dem==5:
                    (mask, withoutMask) = pred
                    label = "Mask" if mask > withoutMask else "No Mask"
                    if (isprocess==False):
                        t = threading.Thread(target=checkk, args=(frame,))
                        t.start()
                cv2.putText(frame, label1, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        RGB = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(np.asarray(RGB)))
        frame2.configure(image=RGB)
        frame2.image = RGB
        cop = label1
        dem+=1
        if dem==6: dem=0
        if (check == True)&( vv== "Video"):
            root.after(7, videoo, qt)
        else:
            isprocess = False
            cv2.destroyAllWindows()
            return
    except:
        pass
def checkcam(qt):

    global isprocess,label,startX, startY, endX, endY,cop,dem,label1,color
    try:
        _, frame = qt.read()
        label1 = cop
        start_time = time.time()
        if (frame.shape[1] > 550) & (frame.shape[1] > frame.shape[0]):
            frame = imutils.resize(frame, width=550)
        elif (frame.shape[0] > 550) & (frame.shape[0] > frame.shape[1]):
            frame = imutils.resize(frame, height=550)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        (locs, preds) = detect_and_predict_mask(frame, net, model,nguong)
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            if (isprocess==False):
                t = threading.Thread(target=checkk, args=(frame,))
                t.start()
            cv2.putText(frame, label1, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        RGB = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(np.asarray(RGB)))
        frame2.configure(image=RGB)
        frame2.image = RGB
        cop = label1
        if (check == True)&( vv == "Camera Live"):
            root.after(80, checkcam, qt)
        else:
            isprocess = False
            cv2.destroyAllWindows()
            return
    except:
        pass
def ex():
    sys.exit()
Button(frame3, text ="Open File",font=("Arial Bold", 13),width=20,height=3,fg='Red', command = openfile).pack(side=LEFT,fill="both", expand="yes")
Button(frame3, text ="Exit",font=("Arial Bold", 13),width=20,height=3,fg='Red', command = ex).pack(side=LEFT,fill="both", expand="yes")
frame1.pack(side=TOP,fill="both", expand="yes")
frame2.pack(fill = BOTH, expand = True)
frame3.pack(side=BOTTOM,fill = BOTH)
root.mainloop()
