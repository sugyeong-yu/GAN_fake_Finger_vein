# GAN_Fake_Finger_vein
GAN을 이용한 fake Fingervein image 생성하기
## 목적 및 필요성
## Protocol
1. 실제 지정맥영상 취득
2. 영상으로부터 frame 취득
3. DCGAN의 입력으로 노이즈 / 실제지정맥 frame을 사용
4. 모델학습
5. 학습된 DCGAN의 Generator로 부터 위조지정맥 이미지 생성
## 모델
- GAN 
  - Discriminator와 Generator 간의 학습이 잘 이루어 지지 않는 문제 발생.
  - 부정확한 이미지 생성 및 loss가 감소하지 않음
  
- DCGAN
  - GAN과 달리 convolution을 사용하는 DCGAN을 사용
![DCGAN구조](https://user-images.githubusercontent.com/70633080/210711727-a730763f-94cd-4be4-b6f7-08ba5ac94a5f.PNG)

## train
|step|epoch|batch_size|learning_rate|D-loss|G-loss|비고|
|---|---|---|---|---|---|---|
|1.|100|64|0.0002|5.3|5.7|epoch-60에서 저장된 모델|

## test
- real 지정맥 이미지\
![image](![image](https://user-images.githubusercontent.com/70633080/210711547-8f780fae-729b-4ebf-a011-b607e97bbb06.png))
- 생성된 fake 지정맥 이미지\
![0](https://user-images.githubusercontent.com/70633080/127965412-eb0e7e87-e849-4638-a28a-69f9a2539869.jpg)
### [참고자료]
- [GAN code(pytorch)](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py)
- [DCGAN code(pytorch)](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
