# LDS U-Net
**Light-Convolution Dense Selection U-Net (LDS U-Net) for Ultrasound Lateral Bony Feature Segmentation**
## LDS U-Net Paper
LDS U-Net was published in Applied Sciences Journal of MDPI under the following citation:

> Banerjee, S.; Lyu, J.; Huang, Z.; Leung, F.H.F.; Lee, T.T.-Y.; Yang, D.; Su, S.; Zheng, Y.; Ling, S.-H. Light-Convolution Dense Selection U-Net (LDS U-Net) for Ultrasound Lateral Bony Feature Segmentation. Appl. Sci. 2021, 11, 10180. https://doi.org/10.3390/app112110180

## Dataset
This model was originally designed to perform the segmentation of ultrasound spine images, but can be used with other datasets as well, for example the dataset from ISIB Challenge available in folder "data" and downloaded from [here](https://github.com/zhixuhao/unet) 

## LDS U-Net Model
LDS U-Net is a lightweight version of U-Net that contains densely connected deptwise separable convolution followed by pointwise convolution, multiscale skip connec-tion, and selection gates. It is inspired by several salient features used in other models  such as  the  U-Net,  MultiResUNet,  and  Attention  U-Net. The architecture of the proposed network of Light-Convolution Dense Selection U-Net (LDS U-Net) is shown below.

![alt text](/images/LDSU-Net.png)

The basic structure of the light dense block is shown below and here, unlike a conventional dense network, all the convolutional layers are depthwise separable convolution layers. The first layer of the light dense block is a depthwise convolution unit that consists of  a  depthwise  convolution  block  followed  by  a  pointwise  convolution  block.  The  next layers are the batch normalization, rectified linear unit (ReLU) activation function, another depthwise convolution unit, and a dropout layer. The first depthwise convolution unit is also connected densely to the dropout layer, as shown by the green dotted arrow in the figure.Through this new design, the light dense block delivers the same advantages as a con-ventional dense block,but with a smaller number of parameters

![alt text](/images/LDSBlock.png)

A multi-scale inception module was incorporated  between  the  encoder  and  decoder  layers  to  enhance  the low-level  features  extracted  in  the  encoder  side.  In the skip path, a sequence of 3×3 convolution  blocks  is  used,  as  shown  in the figure below. The  outputs  from  the  three 3×3 convolution  blocks  are  concatenated  to  enhance the receptive field and reduce the semantic gap between the encoder and the decoder. A  residual  connection  of a 1×1 convolution  block  is  also  presented  in  the  skip path to make the learning procedure stable.

![alt text](/images/SkipPathway.png)







