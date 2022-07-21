# Video_Object_Detection
FAST_RCNN / Yolov 등 여러가지 모델들을 활용한 Video_Object_Detection 구축

## 프로젝트 목적
자동차,사람, 신호등인식 등을 구현하여 OPENCV와 YOLOV5를 기반으로 자율주행 영상처리 웹 서비스를 구현한다.

## 프로젝트 배경
object detection 기술에 대한 이해력 상승.

## 연구 및 개발에 필요한 데이터 셋 소개
1. 자율주행자동차 객체인식 딥러닝 모델을 만들기 위해 동영상링크를 넣어
해당 영상속 차, 사람, 신호등과 같은 객체가 들어가 있는 영상을 사용해야 합니다.

2. 따라서 많은 영상 데이터가 있는
Youtube에서 영상을 검색하여 해당 링크를 사용하였습니다.

https://www.youtube.com/watch?v=Q0Qkqbb_UIU

## 연구 및 개발에 필요한 기술 스택
Yolov5 ⇒ 영상 내에서 지정된 클래스의 객체인식을 해주는 기능
OpenCV ⇒ 영상처리 및 OpenCV SORT기능을 이용한 객체 추적
FLASK ⇒ 딥러닝 기반 영상처리 웹 기능 제작
    - 웹 서비스를 이용하여 사용자가 링크 또는 파일을 업로드할시 딥러닝 모델을
    호출하여 영상처리
    
## 해당 기술(또는 연구방법)의 최근 장점과 단점
- Yolov5를 사용한 장점과 단점
    장점
      → Yolov5는 이미 학습되어있는 모델을 연결하여 객체를 인식해줌
      → Yolov5를 git clone 하여 개인이 직접 커스텀하여 모델을 만들수도 있음
      
      ```Python3
          def load_model(self):
        # YOLOv5 모델 로드
        model = torch.hub.load('ultralytics/yolov5',
                               'yolov5s', pretrained=True)
        model.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
        return model
      ```
      
    단점
      → Yolov5를 이용하여 이미 학습된 모델을 사용하게 되었을 때 지정된 클래스에서 필요한 객체외에 전혀 관련없는 객체를 라벨링하는 경우가 발생함
        
- DeepSORT가 아닌 OpenCV SORT 기능을 사용하는 이유
    DeepSORT는 SORT의 정확도를 개선한 다중 객체 추적기술
    SORT에서 연속된 프레임간의 객체 관계 판별할 때 CNN의 특징맵을 이용하는 특징이 있음
    
    DeepSort의 장점  
      → 정확도를 비약적으로 개선
        
    DeepSort의 단점
      → 필요한 추가 연산의 양이 많아 FPS의 하락요인
        
    
 따라서 자율주행 자동차와 같은 실시간 시스템은 연산속도가 높은 SORT가 더욱 적합하다고 간주되고있다.


## 결과
OD_YOLO.py를 실행시 실시간 Object Detection 영상이 재생되어 상황을 확인시켜주며 완료시 웹 페이지에서 부드럽게 재생이 가능하다.
# 웹페이지
![image](https://user-images.githubusercontent.com/97720878/180149559-80a70487-ebbb-4751-a9ec-faccaace69c0.png)

# 실시간 Object Detection
![image](https://user-images.githubusercontent.com/97720878/180149770-d1009a08-1c1d-452b-9c4b-ca9db64992ba.png)


## 한계점 및 해결 방안
객체에 맞지 않는곳에 박스가 표시되는곳이 많아 Yolov5에서 훈련된 모델이 아닌
연구자가 직접 만든 커스텀 데이터셋을 활용한 모델을 사용하여 객체탐지 모델을  진행해볼 예정.

따라서 더욱 정확한 객체탐지 모델을 구현할 수 있다.
