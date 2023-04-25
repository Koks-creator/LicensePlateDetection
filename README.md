# LicensePlateDetection

Here you can download a model: https://drive.google.com/drive/u/0/folders/1TLZffs4H7tZGroz1Z9DJwCI0R1XN-hLi

I've tried increasing easyocr accuracy with the following steps:
 - making license plates images lerger by scaling
 - adding/removing threshold, it really depends on scenario but I think adding threshold worked well in most cases

Easyocr is overall OCR model that's why training your own OCR model dedicated for this task would perform better
<br>
<br>
Threshold vs no threshold
![comp](https://user-images.githubusercontent.com/73878161/234413979-75654145-32a7-4600-a74f-071a5769e259.png)

Examples
![res1](https://user-images.githubusercontent.com/73878161/234128669-cee3092c-8657-4798-a80a-84409e0b9085.png)
![res3](https://user-images.githubusercontent.com/73878161/234128689-b8ff01eb-6f42-4f24-b01e-4e72774e405e.png)
![res4](https://user-images.githubusercontent.com/73878161/234128702-7439e357-b041-47e5-be86-ebe3b2a6b3a5.png)
![res7](https://user-images.githubusercontent.com/73878161/234414107-ce1072c7-88ac-4d83-86d5-309257a21eb6.png)
![res6](https://user-images.githubusercontent.com/73878161/234414139-5f009537-2931-4878-9fd1-4cd3b8f2a344.png)
![res5](https://user-images.githubusercontent.com/73878161/234414152-ca60883f-5310-4f7b-8d01-8daec75c40ea.png)
