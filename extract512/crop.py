import dlib
import cv2
import os
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("E:/Desktop/samsung/Doan/shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("E:/Desktop/samsung/Doan/dlib_face_recognition_resnet_model_v1.dat")
def training():
    for f in os.listdir('E:/Desktop/samsung/Doan/training_mask'):
        for k in os.listdir('E:/Desktop/samsung/Doan/training_mask/' + f + '/'):
            name_tuple = os.path.splitext(k)
            if(".txt" not in name_tuple[1]):
                img = cv2.imread('E:/Desktop/samsung/Doan/training_mask/' + f + '/'+k)
                #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img1_detection = detector(img, 1)
                print(len(img1_detection))
                if (len(img1_detection) == 1):
                    img1_shape = sp(img, img1_detection[0])
                    img1_aligned = dlib.get_face_chip(img, img1_shape)
                    cv2.imwrite('E:/Desktop/samsung/Doan/training_mask/' + f + '/'+name_tuple[0]+name_tuple[1],img1_aligned)
                else: print("False"+name_tuple[0]+name_tuple[1])


training()
