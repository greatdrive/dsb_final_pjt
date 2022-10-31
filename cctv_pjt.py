import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
#-------default-------
import telegram as tel
import threading
from pygame import mixer
mixer.init() #Initialzing pyamge mixer

mixer.music.load('sober.mp3') #Loading Music File
def send_tele(word):
   bot = tel.Bot(token="5503523524:AAFqRO1K0Xx1fTmI5c6NNWcPp_0Q06BeByU")
   chat_id =  -1001873940891 #5683352342 #5581126225
   image = 'cap.png'
   image2 = 'picture.png'
   bot.sendMessage(chat_id=chat_id, text=word)
   bot.send_photo(chat_id = chat_id, photo=open(image, 'rb'))
   bot.send_photo(chat_id = chat_id, photo=open(image2, 'rb'))

#모델 다운로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best.pt')  # local model
#model = torch.hub.load('C:/Users/Administrator/Desktop/yolov5', 'custom', path='C:/Users/Administrator/Desktop/yolov5/yolov5x6.pt', source='local')  # local repo

#클래스 0번만 적용
model.classes = [0]

# 폰트 색상 지정
blue = (255, 0, 0)
green= (0, 255, 0)
red= (0, 0, 255)
white= (255, 255, 255)
black= (0,0,0)

#FontFace 설정
FontFace =  cv2.FONT_HERSHEY_PLAIN

cap = cv2.VideoCapture(0)
#화면크기 변수
ScreenWidth = 1280
ScreenHeight = 800

#폰트 크기
FontScale = 2
if ScreenWidth<800:
    FontScale = FontScale*0.5
    
#person 위치 변수
PersonScreenWidth = round(ScreenWidth*0.8)
PersonScreenHeight = round(ScreenHeight*0.1)

PersonScreenWidth2 = round(ScreenWidth*0.7)
PersonScreenHeight2 = round(ScreenHeight*0.2)

#화면크기
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ScreenWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ScreenHeight)

#첫 탐지
FirstDetect = False

#사람 탐지 유무
DetectPerson = False

#FrameCount = 0

PointList = []

def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼 Down
        if len(PointList) < 2:
            PointList.append((x, y))

while cap.isOpened():
    ret, frame = cap.read()
    cv2.imwrite('picture.png', frame)
    # Make detections 
    results = model(frame)
    
    #랜더링
    ResultRender =np.squeeze(results.render())
    
    
    if len(PointList) == 2:
        cv2.rectangle(ResultRender, PointList[0], PointList[1], red, 3)
        
        if PointList[0][0] > PointList[1][0]:
            Xmax = PointList[0][0]
            Xmin = PointList[1][0]
        else:
            Xmax = PointList[1][0]
            Xmin = PointList[0][0]
        
        if PointList[0][1] > PointList[1][1]:
            Ymax = PointList[0][1]
            Ymin = PointList[1][1]
        else:
            Ymax = PointList[1][1]
            Ymin = PointList[0][1]
        
    #객체 데이터 프레임
    ResultsDf = results.pandas().xyxy[0]
    #print(ResultsDf)
    
    ResultsDfName = ResultsDf['name'] # 이름 column
    ResultsDfXmin = ResultsDf['xmin'] # xmin column
    ResultsDfXmax = ResultsDf['xmax'] # xmax column
    ResultsDfYmin = ResultsDf['ymin'] # ymin column
    ResultsDfYmax = ResultsDf['ymax'] # ymax column
    
    LenResultsDf = len(ResultsDf)
    PersonCount = 0
    DetectPersonCount = 0
    
    for i in range(LenResultsDf):
        if ResultsDfName[i]=='person':
            PersonCount = PersonCount +1
            PersonX = round(ResultsDfXmin[i]*0.5 + ResultsDfXmax[i]*0.5)
            PersonY = round(ResultsDfYmin[i]*0.5 + ResultsDfYmax[i]*0.5)
            cv2.line(ResultRender, (PersonX, PersonY), (PersonX, PersonY), red, 5)
            if len(PointList) == 2:
                if Xmin <= PersonX <= Xmax:
                    if Ymin <= PersonY <= Ymax:
                        DetectPersonCount = DetectPersonCount + 1
                        if DetectPerson == False:
                            cv2.imwrite('picture.png', frame)
                            cv2.imwrite('cap.png', ResultRender)
                            
                            #os.popen('"./siren2.mp3"')# 경보음 재생    
                            #music.start()              
                            mixer.music.play()                                     
                            #telbot.send_tele(f'경고 : 거수자 {PersonCount}명이 침입')
                            threading.Thread(target=send_tele, args=(f'경고 : 거수자 {PersonCount}명이 침입', )).start()
                            
                        DetectPerson = True
                        

    if cv2.waitKey(10) & 0xFF == ord('r'):
        PointList.clear()
        DetectPerson = False
        mixer.music.stop()
    

        
    if PersonCount>0:
        cv2.putText(ResultRender,f'Person:{PersonCount}', (PersonScreenWidth, PersonScreenHeight), FontFace, FontScale, blue, 1, cv2.LINE_AA)
    
    if DetectPersonCount > 0:
        cv2.putText(ResultRender,f'Detected person:{DetectPersonCount}', (PersonScreenWidth2, PersonScreenHeight2), FontFace, FontScale, white, 1, cv2.LINE_AA)
            
    cv2.imshow('BigBrother', ResultRender)
            
    cv2.setMouseCallback('BigBrother', mouse_handler)
        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
    
        

cap.release()
cv2.destroyAllWindows()

