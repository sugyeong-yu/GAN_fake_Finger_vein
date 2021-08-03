# GAN_fake_Finger_vein
GAN을 이용한 fake Fingervein image 생성하기
- 두가지 모델을 사용해봄
  - GAN, DCGAN
## 모델 설명
- GAN 사용 시 Discriminator와 Generator 간의 학습이 잘 이루어 지지 않음.
  - 부정확한 이미지 생성 및 loss가 감소하지 않는 등 
  - 따라서 GAN과 달리 convolution을 사용하는 DCGAN을 사용
- DCGAN

## train
|step|epoch|batch_size|learning_rate|D-loss|G-loss|비고|
|---|---|---|---|---|---|---|
|1.|100|64|0.0002|5.3|5.7|epoch-60에서 저장된 모델|

## test
- real 지정맥 이미지\
![image](https://user-images.githubusercontent.com/70633080/127966233-167bf36a-d6fa-467d-b3ec-d54971622af7.png)
- 생성된 fake 지정맥 이미지\
![0](https://user-images.githubusercontent.com/70633080/127965412-eb0e7e87-e849-4638-a28a-69f9a2539869.jpg)
### [참고자료]
- [GAN code(pytorch)](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py)
- [DCGAN code(pytorch)](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
