import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

#检测皱纹
def get_wrinkle(image):
    img_yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
    Y,Cr,Cb = cv2.split(img_yuv)
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    avg_gray = np.mean(img_gray[img_gray>0])
    min_gray = np.min(img_gray[img_gray>0])
    #canny = cv2.Canny(img_gray, 50, 150)
    #return canny
    thresold = avg_gray-(avg_gray-min_gray)/2
    _,t0 = cv2.threshold(Y,1,255,0)
    _,t1 = cv2.threshold(Y,thresold,255,1)
    mask = cv2.bitwise_and(t0,t1)
    mask = cv2.merge([mask,mask,mask])
    return mask

#检测痘痘
def get_pimple(image):
    img = image
    img_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    Y,Cr,Cb = cv2.split(img_yuv)
    _,t0 = cv2.threshold(Cb,77,255,0)
    _,t1 = cv2.threshold(Cb,130,255,1)
    mask = cv2.bitwise_and(t0,t1)
    mask = cv2.merge([mask,mask,mask])
    img_pimple = cv2.bitwise_and(img,mask)
    return img_pimple

#检测斑点
#分类层级
def get_spot(image):
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    level = 7 #5个颜色层级
    table = np.zeros((256),dtype=np.uint8)
    for i in range(256):
        lv_size = int(256/level)
        lv = int(i/lv_size)
        lv_value = lv*lv_size
        table[i] = lv_value
    return cv2.LUT(img_gray,table)

# For static images:
IMAGE_FILES = ["E:/PythonApplication1/test4.jpg"]
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            h,w,_ = image.shape
            if(w>h):
                image = cv2.resize(image,(int(256*float(w/h)),256),0,0,0)
            else:
                image = cv2.resize(image,(int(256*float(h/w)),256),0,0,0)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
            #annotated_image = image.copy()
            annotated_image = np.zeros(image.shape,dtype=np.uint8)
            image_rows, image_cols, _ = image.shape
            for face_landmarks in results.multi_face_landmarks:
                #print('face_landmarks:', face_landmarks)
                #pts = []
                idx_to_coordinates = {}
                print("landmark count {}".format(len(face_landmarks.landmark)))
                for idx, landmark in enumerate(face_landmarks.landmark):
                    landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
                    #pts.append([landmark_px[0],landmark_px[1]])
                    if landmark_px:
                        idx_to_coordinates[idx] = landmark_px
                    else:
                        print("not landmark%d"%idx)

                for idx,landmark_px in idx_to_coordinates.items():
                    cv2.circle(annotated_image, landmark_px, drawing_spec.circle_radius,
                               drawing_spec.color, drawing_spec.thickness)
                    cv2.putText(annotated_image,"{}".format(idx),landmark_px,cv2.FONT_HERSHEY_PLAIN,0.8,(0,0,255))

                idx_outlines = [127,234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454,356,389,251,284,332,297,338,10,109,67,103,54,21,162]
                idx_forehead = [9,336,296,334,333,297,297,338,10,109,67,104,105,66,107,9]
                idx_mouth = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61]
                idx_eye_left = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33]
                idx_eye_right = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]
                idx_canthus_left = [226,124,156,143,111,31,226]
                idx_canthus_right = [446,353,383,372,340,261,446]
                idx_nose = [6,196,236,134,131,48,64,240,97,2,326,460,294,278,360,363,456,419,6]
                idx_cheek_left = [129,209,126,47,121,231,230,229,228,31,111,116,123,147,213,192,214,212,216,206,203,129]
                idx_cheek_right = [358,429,355,277,349,450,449,448,261,340,345,352,376,433,416,434,432,436,426,423,358]

                pts_outlines = np.asarray([idx_to_coordinates[idx] for idx in idx_outlines ],np.int32)
                pts_forehead = np.asarray([idx_to_coordinates[idx] for idx in idx_forehead ],np.int32)
                pts_mouth = np.asarray([idx_to_coordinates[idx] for idx in idx_mouth ],np.int32)
                pts_eye_left = np.asarray([idx_to_coordinates[idx] for idx in idx_eye_left ],np.int32)
                pts_eye_right = np.asarray([idx_to_coordinates[idx] for idx in idx_eye_right ],np.int32)
                pts_canthus_left = np.asarray([idx_to_coordinates[idx] for idx in idx_canthus_left ],np.int32)
                pts_canthus_right = np.asarray([idx_to_coordinates[idx] for idx in idx_canthus_right ],np.int32)
                pts_nose = np.asarray([idx_to_coordinates[idx] for idx in idx_nose ],np.int32)
                pts_cheek_left = np.asarray([idx_to_coordinates[idx] for idx in idx_cheek_left ],np.int32)
                pts_cheek_right = np.asarray([idx_to_coordinates[idx] for idx in idx_cheek_right ],np.int32)

                mask_outlines = np.zeros(image.shape,dtype=np.uint8)
                cv2.fillPoly(mask_outlines,[pts_outlines],(255,255,255))
                mask_forehead = np.zeros(image.shape,dtype=np.uint8)
                cv2.fillPoly(mask_forehead,[pts_forehead],(255,255,255))
                mask_mouth = np.zeros(image.shape,dtype=np.uint8)
                cv2.fillPoly(mask_mouth,[pts_mouth],(255,255,255))
                mask_eye = np.zeros(image.shape,dtype=np.uint8)
                cv2.fillPoly(mask_eye,[pts_eye_left,pts_eye_right],(255,255,255))
                mask_canthus = np.zeros(image.shape,dtype=np.uint8)
                cv2.fillPoly(mask_canthus,[pts_canthus_left,pts_canthus_right],(255,255,255))
                mask_nose = np.zeros(image.shape,dtype=np.uint8)
                cv2.fillPoly(mask_nose,[pts_nose],(255,255,255))
                mask_cheek = np.zeros(image.shape,dtype=np.uint8)
                cv2.fillPoly(mask_cheek,[pts_cheek_left,pts_cheek_right],(255,255,255))


                #mask_img = get_mask(np.zeros(mask_forehead.shape,dtype=np.uint8),[mask_forehead,mask_canthus])
                #mask_img = mask_img/3
                #mask_img = np.asarray(mask_img,dtype=np.uint8)
                #cv2.imshow("area",cv2.add(image,mask_img))
                #cv2.imshow("outlines",cv2.bitwise_and(mask_outlines,image))
                #cv2.imshow("eye",cv2.bitwise_and(mask_eye,image))

                #img_forehead = cv2.bitwise_and(mask_forehead,image)
                
                #img_skin = get_skin(image,mask_outlines,mask_eye,mask_mouth)
                #cv2.imshow("skin",img_skin)

                mask = cv2.bitwise_or(mask_forehead,mask_canthus)
                image_wrinkle_area = cv2.bitwise_and(image,mask)
                image_wrinkle_mask = get_wrinkle(image_wrinkle_area)
                image_wrinkle_mask = np.asarray(image_wrinkle_mask/3,dtype=np.uint8)
                cv2.imshow("wrinkle",cv2.add(image,image_wrinkle_mask)) #皱纹

                image_face = cv2.bitwise_and(image,mask_outlines)
                image_face = cv2.subtract(image_face,mask_eye)
                image_face = cv2.subtract(image_face,mask_mouth)

                #cv2.imshow("pimple",get_pimple(image_face)) #痘痘
                #cv2.imshow("spot",get_spot(image_face)) #斑点

                '''
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                '''
            #cv2.imshow("annotated_image",annotated_image)
            cv2.waitKey()
            #cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)