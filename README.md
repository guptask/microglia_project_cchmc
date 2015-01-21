#Microglia Segmentation Project

##Packages to install

+ ###opencv: 
>Clone the opencv repo at https://github.com/Itseez/opencv. Follow the 
instructions for installation on Linux at http://opencv.org/. Add to 
~/.bashrc **export LD\_LIBRARY\_PATH=${LD\_LIBRARY\_PATH}:/usr/local/lib**. 
To test sample opencv code, compile using 
**g++ <file\_name> `pkg-config opencv --cflags --libs`**


##Build and run microglia segmentation package

Inside the project root directory, type **make** to build the project.
A binary called **segment** will be created.

Command to run the software: 
**./segment <image directory with / at end> <image list> <error file> 
<output csv file>**

