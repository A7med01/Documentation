# Size and color sorting  - version one 

[image1]: ./image1_1.jpg "first"
[image2]: ./image.png "second"

![alt text][image1]

This code give the color, diameter and shape of an fruit :
- we need the background to be white so we used a white box as shown  above
- industrial camera on the top of the box and light source as shown above
- first we will take picture using the camera then detect the fruit shape usning classical computer vision techniques , then using this shape we can get the diameter and the color of this area as shown below

![alt text][image2]


Tools :
- python 3
- OpenCV 4

Detaied discription of the code in the note book 
Notes : 
- Simple and fast cod  (speed is about 50 image per second)
- Work bad if we change the backgroun color or change the light source position , it will work as long as the code is customize with the background and the light source and itensity and this is the case in our application 

# This notebook has code with discription:[Here](https://github.com/A7med01/Documentation/blob/master/size_color_fruits_v1.ipynb)
