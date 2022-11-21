import cv2                                         #本開開發測試系統參閱於https://steam.oxxostudio.tw/category/python/index.html
import numpy as np
detector = cv2.CascadeClassifier('lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml')  # 載入人臉追蹤模型
recog = cv2.face.LBPHFaceRecognizer_create()      # 啟用訓練人臉模型方法
faces = []   # 儲存人臉位置大小的串列
ids = []     # 記錄該人臉 id 的串列

for i in range(1,20):
    img = cv2.imread(f'face01/{i}.jpg')           # 依序開啟每一張人臉的照片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
    img_np = np.array(gray,'uint8')               # 轉換成指定編碼的 numpy 陣列
    face = detector.detectMultiScale(gray)        # 擷取人臉區域
    for(x,y,w,h) in face:
        faces.append(img_np[y:y+h,x:x+w])         # 記錄1號人臉的位置和大小內像素的數值
        ids.append(1)                             # 記錄人臉對應的 id，只能是整數，都是 1 表示一號人臉的 id 為 1

for i in range(1,4):
    img = cv2.imread(f'face02/{i}.jpg')           # 依序開啟每一張人臉的照片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
    img_np = np.array(gray,'uint8')               # 轉換成指定編碼的 numpy 陣列
    face = detector.detectMultiScale(gray)        # 擷取人臉區域
    for(x,y,w,h) in face:
        faces.append(img_np[y:y+h,x:x+w])         # 記錄二號人臉的位置和大小內像素的數值
        ids.append(2)                             # 記錄二號人臉對應的 id，只能是整數，都是 2 表示二號人臉的 id 為 2
                                   
for i in range(1,4):
    img = cv2.imread(f'face03/{i}.jpg')           # 依序開啟每一張人臉的照片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
    img_np = np.array(gray,'uint8')               # 轉換成指定編碼的 numpy 陣列
    face = detector.detectMultiScale(gray)        # 擷取人臉區域
    for(x,y,w,h) in face:
        faces.append(img_np[y:y+h,x:x+w])         # 記錄三號人臉的位置和大小內像素的數值
        ids.append(3)                             # 記錄三號人臉對應的 id，只能是整數，都是 3 表示三號人臉的 id 為 3

print('訓練開始...')                              # 提示開始訓練
recog.train(faces,np.array(ids))                  # 開始訓練
recog.save('face.yml')                            # 訓練完成儲存為 face.yml
print('ok!訓練完成，輸出yml檔案')
