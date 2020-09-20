import cv2
import numpy as np
from PIL import Image
from keras.models import model_from_json
import webbrowser
import os
import time
import matplotlib.pyplot as plt
# import subprocess
import sendKeys_HMI as sK
# import mnist_handrecog
drawing = False
sign_lang_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 11: 'L', 12: 'M',
                  13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
                  24: 'Y'}
point = (0, 0)
img_counter = 0
image_sz = 200
# firefox_path = "C:/Program Files/Mozilla Firefox/firefox.exe %s"

json_file = open('H_Cl_model_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("H_Cl_model_weights_1.h5")
print("Loaded model from disk")


def mouse_drawing(event, x, y, flags, params):
    global point, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        point = (x, y)


cap = cv2.VideoCapture(0)
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_drawing)

while True:
    _, frame = cap.read()
    if drawing:
        #       Rectangle(img, start_pt, end_pt, (B,G,R), thickness)
        cv2.rectangle(frame, point, (point[0] + image_sz, point[1] + image_sz), (128, 128, 0), 1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(25)
    if key == 32:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        img = Image.open(img_name).convert('L')
        img_crop = img.crop((point[0], point[1], point[0] + image_sz, point[1] + image_sz))
        img_crop = img_crop.resize((28, 28), Image.ANTIALIAS)
        # img_crop.save('H_CL_{}.png'.format(img_counter))
        cap.release()
        #         im2 = copy.deepcopy(img_crop)
        print(type(img_crop))
        if key == 32:
            cap.release()
            break
    elif key == 27:
        cap.release()
        break

# print(img_crop.shape)
# plt.imshow(img_crop, cmap='binary_r')
# plt.show()
# cap.release()
cv2.destroyAllWindows()

img_arr = np.array(img_crop)
img_arr = img_arr.reshape(1, 28, 28, 1)
# loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adaDelta', metrics=['accuracy'])
pred = loaded_model.predict_classes(img_arr)
print(pred)


for a in pred:
    ret_val = a
# ret_val = [x for x in pred]
# print(ret_val, "sdsgdg")
# for a in sign_lang_dict:

if ret_val == 0:
    os.system("start ms-settings:")
    # print(sign_lang_dict[0], "predicted")
elif ret_val == 1:
    webbrowser.get().open("http://google.com")
    # print(sign_lang_dict[1], "predicted")
elif ret_val == 2:
    os.startfile('microsoft.windows.camera:')
    # subprocess.run('start microsoft.windows.camera', shell=True)
    # print(sign_lang_dict[1], "predicted")
elif ret_val == 3:
    sK.bringDown()
    time.sleep(1)
    sK.refresh()
elif ret_val == 5:
    sK.newTab()
elif ret_val == 14:
    sK.taskManager()
elif ret_val == 15:
    sK.tabChangeScrShot()
else:
    print("wrong!")

print(sign_lang_dict[ret_val], "classified!")