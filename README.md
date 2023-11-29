# 한밭대학교 컴퓨터공학과 Andrew Ng팀

**팀 구성**
- 20201747 이상욱 
- 20191781 이가현
- 20201774 신유정

## <u>Teamate</u> Project Background
- ### 필요성
  - 돼지의 번식은 양돈 농가의 수익과 직결되기 때문에 돼지의 출산율을 높이는 것은 중요하다. 출산율을 높이기 위해 적절한 시기에 교배를 해야 하며, 교배를 위해 인공수정 기법이 주로 사용된다.
  - 성공적인 인공수정을 위해서는 돼지의 발정 여부를 확인하여 적절한 시점을 찾아야 하며, 적절한 시점을 찾기 위해 웅돈을 교배사로 데려와 웅돈을 모돈에게 노출한 뒤, 모돈의 반응행동을 확인하기 위해 등에 올라타보는 등 많은 작업을 필요로 한다.
  - 또한 농장의 업무 시간 외에는 발정 시기를 확인할 수 없기 때문에 적절한 시기에 인공수정을 하기 위해서는 많은 시간과 노동력이 요구된다.
  - 우리는 일련의 발정 체크 과정을 간소화 하여 농장의 효율성을 높이기 위해 딥러닝 기법으로 돼지들의 행동 패턴을 분석하여 인공수정 시점을 예측하고자 한다.
- ### 기존 해결책의 문제점
  - 기존에는 돼지의 발정을 체크하기 위해 열화상 카메라와 센서 등을 이용하였지만 이러한 방법들은 단가가 비싸고 유지보수가 힘들다는 단점이 존재한다.
  - 귀를 쫑긋 피는 행위를 제외한 다른 돼지의 발정 징후(활동량 증가 등)들은 육안으로 알아보기 힘들다.
  
## System Design
  - ### System Requirements
    <img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black"/> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white"/> <img src="https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white"/> <img src="https://img.shields.io/badge/Flutter-02569B?style=for-the-badge&logo=flutter&logoColor=white"/> <img src="https://img.shields.io/badge/MySQL-00000F?style=for-the-badge&logo=mysql&logoColor=white"/> <img src="https://img.shields.io/badge/nginx-%23009639.svg?style=for-the-badge&logo=nginx&logoColor=white"/>
    
    - DL Server
      - cuda==12.1
      - torch==2.0.0
      - torchvision==0.15.1
      - opencv-python==4.7.0.72
      - numpy==1.24.2
      - pandas==2.0.0
      - ultralytics==8.0.72
      - tqdm==4.65.0

    - API Server
      - Django==4.2.6
      - django-cors-headers==4.3.0
      - djangorestframework==3.14.0
      - gunicorn==20.1.0

    - DB Server
      - mysql-connector-python==8.2.0
      - mysqlclient==2.1.1
   
  - ### System Architecture
    <img width="1000" alt="스크린샷 2023-10-27 오후 5 03 28" src="https://github.com/HBNU-SWUNIV/come-capstone23-andrew-ng/assets/83907194/6374ad0e-eca5-4781-9c9e-eea18b12c238">

  - ### System Process
    <img width="1000" alt="KakaoTalk_Photo_2023-10-27-22-42-27" src="https://github.com/HBNU-SWUNIV/come-capstone23-andrew-ng/assets/83907194/3ea01f56-081e-479a-ad16-0a0e7be70ca3">

  - ### Deep Learning Flow Chart
    <img width="350" alt="flow chart" src="https://github.com/HBNU-SWUNIV/come-capstone23-andrew-ng/assets/83907194/08db7880-46d8-4b4a-b4b3-1b1194012dfc">


  
  
## Results
  - ### 접속 화면 (노트 접속)
  - https://github.com/HBNU-SWUNIV/come-capstone23-andrew-ng/issues/1#issue-2016462613
  - https://github.com/HBNU-SWUNIV/come-capstone23-andrew-ng/issues/1
    
  - ### 접속 화면 (핸드폰 접속)
  - https://github.com/HBNU-SWUNIV/come-capstone23-andrew-ng/issues/2#issue-2016468592
  - https://github.com/HBNU-SWUNIV/come-capstone23-andrew-ng/issues/2
