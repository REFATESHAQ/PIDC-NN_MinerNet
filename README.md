# The architecture of the proposed PIDC-NN, also called MinerNet. To read and download the article, please click here [PIDC-NN (MinerNet) Article](https://doi.org/10.36227/techrxiv.23266301.v3)
![Drawing2](https://github.com/REFATESHAQ/PIDC-NN_MinerNet/assets/48349737/2a9502ef-5f2b-4dbf-b89d-c19c90201021)
![Loss](https://github.com/REFATESHAQ/PIDC-NN_MinerNet/assets/48349737/09ffd34c-d2b9-407b-92c2-28d63cfd226b)
![PIDC-NN](https://github.com/REFATESHAQ/PIDC-NN_MinerNet/assets/48349737/22a72763-ed25-4636-80ce-ed1da0314f2a)

### To Whom It May Concern
### Members of the Scientific Community


Dear colleagues:

I hope my letter finds you well. My name is REFAT MOHAMMED ABDULLAH ESHAQ (https://orcid.org/0000-0002-6448-4054). I have created a new algorithm, namely Proportional–Integral–Derivative–Cumulative Neural Networks (PIDC-NN), also called MinerNet. This algorithm work based on the PID controller that was created by the inventor Elmer Sperry in 1910. The code has been released on GitHub, see https://github.com/REFATESHAQ/PIDC-NN_MinerNet , and https://github.com/REFATESHAQ/PIDC-NN_MinerNet-Pro . The data (Coal and Gangue Infrared Images in BMP file format) has been released on IEEE Dataport. https://dx.doi.org/10.21227/v3m7-dk11

Although convolutional neural networks (CNNs) have achieved great successes in computer vision and pattern recognition, they have some shortcomings. In this article, a novel deep learning algorithm for binary classification is proposed to distinguish between coal and gangue infrared images. First, a Proportional–Integral–Derivative–Cumulative (PIDC) algorithm is created, which works based on the concept of a PID controller, in order to quickly extract features from infrared images and also to control the performance of Artificial Neural Networks (ANNs). Second, an ANN is designed for binary classification tasks (coal/gangue). Third, the PIDC algorithm and the ANN algorithm are connected to create a new learning system, namely, the Proportional–Integral–Derivative–Cumulative Neural Network (PIDC-NN), also called MinerNet. The proposed PIDC-NN architecture works without any traditional layers of deep CNNs such as convolutional layers, nonlinear activation functions layers, batch normalization layers, polling layers, or dropout layers. The results of the training and test processes demonstrate that the proposed PIDC-NN architecture alleviates the oscillation and overfitting problems of existing CNNs. Moreover, it solves the problem of dead neurons and big data that are required to train CNNs. Additionally, it provides robust and resilient control by tuning the gain coefficients KP, KI, and KD; the sampling time (dt); and arbitrary value (AV). A comparison between the proposed PIDC-NN architecture and state-of-the-art CNNs proves the effectiveness of the proposed method in accelerating both the training and test processes with competitive loss and accuracy. 

I emphasize that this algorithm (PIDC) that I created through my own effort, can provide optimal control to any system (not only ANN) whether linear or nonlinear with multiple inputs. Furthermore, this algorithm (PIDC) can control multiple complicated random inputs and make the system linear even with inputs, their amounts, and values are huge numbers (goes to infinity). 

The code is licensed under GNU Affero General Public License Version 3 (GNU AGPLv3); for more information, see https://www.gnu.org/licenses/agpl-3.0.en.html. The dataset (Coal and Gangue Infrared Images in BMP file format (Data.zip)) is licensed under a  Creative Commons Attribution 4.0 International (CC BY 4.0) License. For more information, see https://creativecommons.org/licenses/by/4.0/. 
This work has been supported by my livelihood and my family's aid. The code and data are connected to article, entitled “Deep Learning Algorithm for Computer Vision with a New Technique and Concept: PIDC-NN for Binary Classification Tasks in a Coal Preparation Plant (MinerNet)” TechRxiv, see, https://doi.org/10.36227/techrxiv.23266301.v3 


# Important Notice

If you would like to operate this Network with other images in any other field but the number of images is different, you must adjust these numbers as shown in the below figure. For example, in my case, the number of infrared images of coal was 308 images and the number of infrared images of gangue was 308 images so the total was 616  images. The numbers of training, validation, and test images, in the Data processing section of the code, as displayed in the below figure,  were adjusted based on the number of images I had. If you do not understand what I mean and do not adjust these numbers, certainly the Network does not work well.  

![Capture](https://github.com/REFATESHAQ/PIDC-NN_MinerNet/assets/48349737/056c7d6c-5dfc-4dd4-9c6b-e095c2b0151c)
