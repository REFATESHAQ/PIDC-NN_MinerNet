%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Algorithm Name: Proportional–Integral–Derivative–Cumulative Neural Network (PIDC-NN), also called (MinerNet).
% Written By: Refat Mohammed Abdullah Eshaq, on 30/8/2022.
% Copyright (c) 2022, Refat Mohammed Abdullah Eshaq, All rights reserved.
% This code is licensed under a GNU Affero General Public License Version 3 (GNU AGPLv3), for more information, see <https://www.gnu.org/licenses/agpl-3.0.en.html>.
% This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
% This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details. You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
% This work has been supported by my livelihood and my family's aid. 
% The code and data are connected to the article, entitled “Deep Learning Algorithm for Computer Vision with a New Technique and Concept: PIDC-NN for Binary Classification Tasks in a Coal Preparation Plant (MinerNet)”, see  <https://www.techrxiv.org/articles/preprint/Deep_Learning_Algorithm_for_Computer_Vision_with_a_New_Technique_and_Concept_PIDC-NN_for_Binary_Classification_Tasks_in_a_Coal_Preparation_Plant_MinerNet_/23266301/2>. Note that, the article is under review. 
% Publish Date: 2023-06-06.  
% Author's Email: refateshaq1993@gmail.com; refateshaq@hotmail.com; fs18050005@cumt.edu.cn;  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%#####################################################################################################################################################################################################################################%
 %% Important Notices: 
 % If you want to download the data (Coal and Gangue Infrared Images in BMP files format (Data.zip)), please go to this website, IEEE Dataport <https://dx.doi.org/10.21227/v3m7-dk11>. and download this file "Data.zip"; this zip file includes four folders ("Coal Original Infrared Images", "Gangue Original Infrared images", "Coal_resize(240_54)", and "Gangue_resize(240_54)").
 % "Coal Original Infrared Images" folder and "Gangue Original Infrared Images" folder contain the images that I obtained from the infrared camera (rows * columns * channels)(240*320*3) before the resizing).
 % In order to use this PIDC-NN algorithm (MinerNet) the infrared images or any other type of images should be in size (240*54*3), so there are two folders in the name of "Coal_resize(240_54)" and "Gangue_resize(240_54)". These two folders should be only used as the data input for binary classification.
 % Please run the code twice. You must not take the results of PIDC-NN (MinerNet) from the first run.

 %% Code starts from here.
tic 
clc;  % Clear the command window.
clear all;
close all;

folder1 = 'C:\Users\HP\Documents\MATLAB\Neural Network (NN)\Data\Coal_resize(240_54)';    %You can define the path of a folder 1 to access the images of coal. 
folder2 = 'C:\Users\HP\Documents\MATLAB\Neural Network (NN)\Data\Gangue_resize(240_54)';  % You can define the path of a folder 2 to access the images of gangue. 

fileInfo1 = dir(fullfile(folder1, '*.bmp'));
numberOfImageFiles1 = length(fileInfo1)
fileInfo2 = dir(fullfile(folder2, '*.bmp'));
numberOfImageFiles2 = length(fileInfo2)
total_images = numberOfImageFiles1 + numberOfImageFiles2

load_images = cell(total_images, 2);

% Load folder 1 images into first part of cell array.
for r = 1: length(fileInfo1);
  thisFilename = fullfile(fileInfo1(r).folder, fileInfo1(r).name);
  load_images{r,1} = 1;
  load_images{r,2} = imread(thisFilename); 
end

% Now follow on by loading folder 2 images into latter part of cell array.
for r = 1 : length(fileInfo2);
  thisFilename = fullfile(fileInfo2(r).folder, fileInfo2(r).name);
  load_images{length(fileInfo1)+r,1} = 2;
  load_images{length(fileInfo1)+r,2} = imread(thisFilename);
end
                                       %%% End images load %%%
%% Feature extraction process starts
Final_Output = zeros (616, 240);
Kp = 1*10^-6;          % proportional (KP)
Ki = 5*10^-6;          % Integral (KI)
Kd = 5*10^-6;          % derivative (KD)
dt = 0.1;              % sampling time (s)
ArbitraryValue = 10;

for chanal = 1:3
for data = 1: length(load_images)
    
Image = load_images{data,2};
x = double(Image);
x= x(:,:,chanal);

number_of_neuron =  size(x,1); 

Error(1:number_of_neuron) = 0;
FeedBack(1:number_of_neuron)=0;

for  epochPID = 1:8

for i = 1:number_of_neuron
    
 Error(i+1) = ArbitraryValue - FeedBack(i);
 % error entering the PID controller
 
   
                       %%%%%%%%%% Features x1, x2, and x3 %%%%%%%%%%
                             %%%%%%%%%% PID Unit (1)%%%%%%%%%%
    Prop_1(i) = x(i,1).*Error(i+1);% error of proportional term
    Der_1(i)  = x(i,2).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_1(i)  = x(i,3).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_1(i)    = sum(Int_1); % the sum of the integration of the error
    
    PID_1(i)  = Kp*Prop_1(i) + Ki* SumInt_1(i)+ Kd*Der_1(i); % the three PID terms

                       %%%%%%%%%% Features x4, x5, and x6 %%%%%%%%%%
                             %%%%%%%%%% PID Unit (2)%%%%%%%%%%
    Prop_2(i) = x(i,4).*Error(i+1);% error of proportional term
    Der_2(i)  = x(i,5).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_2(i)  = x(i,6).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_2(i) = sum(Int_2); % the sum of the integration of the error
    
    PID_2(i)  = Kp*Prop_2(i) + Ki*SumInt_2(i)+ Kd*Der_2(i); % the three PID terms
 
                       %%%%%%%%%% Features x7, x8, and x9 %%%%%%%%%%
                             %%%%%%%%%% PID Unit (3)%%%%%%%%%%
    Prop_3(i) = x(i,7).*Error(i+1);% error of proportional term
    Der_3(i)  = x(i,8).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_3(i)  = x(i,9).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_3(i) = sum(Int_3); % the sum of the integration of the error
    
    PID_3(i)  = Kp*Prop_3(i) + Ki*SumInt_3(i)+ Kd*Der_3(i); % the three PID terms

                 %%%%%%%%%% Sum of outputs of PID Unit(1),(2),and (3)%%%%%%%%%%
    Sum_Out_PID_1(i) = PID_1(i) + PID_2(i) + PID_3(i); 
    
                 %%%%%%%%%% Sum of outputs entering the PID Sum Unit (1)
    Prop_Sum_1(i) =  Sum_Out_PID_1(i).*Error(i+1);% error of proportional term
    Der_Sum_1(i)  =  Sum_Out_PID_1(i).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_Sum_1(i)  =  Sum_Out_PID_1(i).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_Sum_1(i) = sum(Int_Sum_1); % the sum of the integration of the error
    
    PID_Sum_1(i) = Kp*Prop_Sum_1(i) + Ki*SumInt_Sum_1(i)  + Kd*Der_Sum_1(i);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
                       %%%%%%%%%% Features x10, x11, and x12 %%%%%%%%%%
                               %%%%%%%%%% PID Unit (4)%%%%%%%%%%
    Prop_4(i) = x(i,10).*Error(i+1);% error of proportional term
    Der_4(i)  = x(i,11).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_4(i)  = x(i,12).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_4(i) = sum(Int_4); % the sum of the integration of the error
    
    PID_4(i)  = Kp*Prop_4(i) + Ki*SumInt_4(i)+ Kd*Der_4(i); % the three PID terms

                       %%%%%%%%%% Features x13, x14, and x15 %%%%%%%%%%
                               %%%%%%%%%% PID Unit (5)%%%%%%%%%%
    Prop_5(i) = x(i,13).*Error(i+1);% error of proportional term
    Der_5(i)  = x(i,14).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_5(i)  = x(i,15).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_5(i) = sum(Int_5); % the sum of the integration of the error
    
    PID_5(i)  = Kp*Prop_5(i) + Ki*SumInt_5(i)+ Kd*Der_5(i); % the three PID terms

                      %%%%%%%%%% Features x16, x17, and x18 %%%%%%%%%%
                              %%%%%%%%%% PID Unit (6)%%%%%%%%%%
    Prop_6(i) = x(i,16).*Error(i+1);% error of proportional term
    Der_6(i)  = x(i,17).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_6(i)  = x(i,18).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_6(i) = sum(Int_6); % the sum of the integration of the error
    
    PID_6(i) = Kp*Prop_6(i) + Ki*SumInt_6(i)+ Kd*Der_6(i); % the three PID terms

                       %%%%%%%%%% Sum of PID Unit(4),(5),and (6)%%%%%%%%%%
    Sum_Out_PID_2(i) = PID_4(i) + PID_5(i) + PID_6(i);
    
                       %%%%%%%%%% Output entering the PID Sum Unit (2) 
    Prop_Sum_2(i) =  Sum_Out_PID_2(i).*Error(i+1);% error of proportional term
    Der_Sum_2(i)  =  Sum_Out_PID_2(i).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_Sum_2(i)  =  Sum_Out_PID_2(i).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_Sum_2(i) = sum(Int_Sum_2); % the sum of the integration of the error
    
    PID_Sum_2(i) = Kp*Prop_Sum_2(i) + Ki*SumInt_Sum_2(i)  + Kd*Der_Sum_2(i);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                     %%%%%%%%%% Features x19, x20, and x21 %%%%%%%%%%
                             %%%%%%%%%% PID Unit (7)%%%%%%%%%%
    Prop_7(i) = x(i,19).*Error(i+1);% error of proportional term
    Der_7(i)  = x(i,20).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_7(i)  = x(i,21).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_7(i) = sum(Int_7); % the sum of the integration of the error
    
    PID_7(i) = Kp*Prop_7(i) + Ki*SumInt_7(i)+ Kd*Der_7(i); % the three PID terms
  
                     %%%%%%%%%% Features x22, x23, and x24 %%%%%%%%%%
                             %%%%%%%%%% PID Unit (8)%%%%%%%%%%
    Prop_8(i) = x(i,22).*Error(i+1);% error of proportional term
    Der_8(i)  = x(i,23).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_8(i)  = x(i,24).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_8(i) = sum(Int_8); % the sum of the integration of the error
    
    PID_8(i) = Kp*Prop_8(i) + Ki*SumInt_8(i)+ Kd*Der_8(i); % the three PID terms
   
                     %%%%%%%%%% Features x25, x26, and x27 %%%%%%%%%%
                            %%%%%%%%%% PID Unit (9)%%%%%%%%%%
    Prop_9(i) = x(i,25).*Error(i+1);% error of proportional term
    Der_9(i)  = x(i,26).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_9(i)  = x(i,27).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_9(i) = sum(Int_9); % the sum of the integration of the error
    
    PID_9(i) = Kp*Prop_9(i) + Ki*SumInt_9(i)+ Kd*Der_9(i); % the three PID terms
 
                %%%%%%%%%% Sum of outputs PID Unit(7),(8),and (9)%%%%%%%%%%
    Sum_Out_PID_3(i) = PID_7(i) + PID_8(i) + PID_9(i);
    
              %%%%%%%%%% Sum of outputs entering the PID Sum Unit (3) 
    Prop_Sum_3(i) =  Sum_Out_PID_3(i).*Error(i+1);% error of proportional term
    Der_Sum_3(i)  =  Sum_Out_PID_3(i).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_Sum_3(i)  =  Sum_Out_PID_3(i).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_Sum_3(i) = sum(Int_Sum_3); % the sum of the integration of the error
    
     PID_Sum_3(i)  = Kp*Prop_Sum_3(i) + Ki*SumInt_Sum_3(i) + Kd*Der_Sum_3(i);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    PID_Total_1(i) = PID_Sum_1(i) + PID_Sum_2(i) + PID_Sum_3(i);
     
    Prop_Total_1(i) =  PID_Total_1(i).*Error(i+1);% error of proportional term
    Der_Total_1(i)  =  PID_Total_1(i).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_Total_1(i)  =  PID_Total_1(i).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_Total_1(i) = sum(Int_Total_1); % the sum of the integration of the error
    
    PID_Last_Uint_1(i) = Kp*Prop_Total_1(i) + Ki*SumInt_Total_1(i) + Kd*Der_Total_1(i);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                      %%%%%%%%%% Features x28, x29, and x30 %%%%%%%%%%
                             %%%%%%%%%% PID Unit (10)%%%%%%%%%%
    Prop_10(i) = x(i,28).*Error(i+1);% error of proportional term
    Der_10(i)  = x(i,29).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_10(i)  = x(i,30).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_10(i) = sum(Int_10); % the sum of the integration of the error
    
    PID_10(i)  = Kp*Prop_10(i) + Ki* SumInt_10(i)+ Kd*Der_10(i); % the three PID terms

                      %%%%%%%%%% Features x31, x32, and x33 %%%%%%%%%%
                             %%%%%%%%%% PID Unit (11)%%%%%%%%%%
    Prop_11(i) = x(i,31).*Error(i+1);% error of proportional term
    Der_11(i)  = x(i,32).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_11(i)  = x(i,33).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_11(i) = sum(Int_11); % the sum of the integration of the error
    
    PID_11(i)  = Kp*Prop_11(i) + Ki*SumInt_11(i)+ Kd*Der_11(i); % the three PID terms
 
                      %%%%%%%%%% Features x34, x35, and x36 %%%%%%%%%%
                             %%%%%%%%%% PID Unit (12)%%%%%%%%%%
    Prop_12(i) = x(i,34).*Error(i+1);% error of proportional term
    Der_12(i)  = x(i,35).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_12(i)  = x(i,36).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_12(i) = sum(Int_12); % the sum of the integration of the error
    
    PID_12(i)  = Kp*Prop_12(i) + Ki*SumInt_12(i)+ Kd*Der_12(i); % the three PID terms

                %%%%%%%%%% Sum of outputs of PID Unit(10),(11),and (12)%%%%%%%%%%
    Sum_Out_PID_4(i) = PID_10(i) + PID_11(i) + PID_12(i); 
    
                %%%%%%%%%% Sum of outputs entering the PID Sum Unit (4)
    Prop_Sum_4(i) =  Sum_Out_PID_4(i).*Error(i+1);% error of proportional term
    Der_Sum_4(i)  =  Sum_Out_PID_4(i).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_Sum_4(i)  =  Sum_Out_PID_4(i).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_Sum_4(i) = sum(Int_Sum_4); % the sum of the integration of the error
    
    PID_Sum_4(i)  = Kp*Prop_Sum_4(i) + Ki*SumInt_Sum_4(i)  + Kd*Der_Sum_4(i);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
                     %%%%%%%%%% Features x37, x38, and x39 %%%%%%%%%%
                             %%%%%%%%%% PID Unit (13)%%%%%%%%%%
    Prop_13(i) = x(i,37).*Error(i+1);% error of proportional term
    Der_13(i)  = x(i,38).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_13(i)  = x(i,39).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_13(i) = sum(Int_13); % the sum of the integration of the error
    
    PID_13(i)  = Kp*Prop_13(i) + Ki*   SumInt_13(i)+ Kd*Der_13(i); % the three PID terms

                     %%%%%%%%%% Features x40, x41, and x42 %%%%%%%%%%
                             %%%%%%%%%% PID Unit (14)%%%%%%%%%%
    Prop_14(i) = x(i,40).*Error(i+1);% error of proportional term
    Der_14(i)  = x(i,41).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_14(i)  = x(i,42).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_14(i) = sum(Int_14); % the sum of the integration of the error
    
    PID_14(i)  = Kp*Prop_14(i) + Ki*SumInt_14(i)+ Kd*Der_14(i); % the three PID terms

                     %%%%%%%%%% Features x43, x44, and x45 %%%%%%%%%%
                             %%%%%%%%%% PID Unit (15)%%%%%%%%%%
    Prop_15(i) = x(i,43).*Error(i+1);% error of proportional term
    Der_15(i)  = x(i,44).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_15(i)  = x(i,45).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_15(i) = sum(Int_15); % the sum of the integration of the error
    
    PID_15(i)  = Kp*Prop_15(i) + Ki*SumInt_15(i)+ Kd*Der_15(i); % the three PID terms

              %%%%%%%%%% Sum of PID Unit(13),(14),and (15)%%%%%%%%%%
    Sum_Out_PID_5(i) = PID_13(i) + PID_14(i) + PID_15(i);
    
                %%%%%%%%%% Output entering the PID Sum Unit (5) 
    Prop_Sum_5(i) =  Sum_Out_PID_5(i).*Error(i+1);% error of proportional term
    Der_Sum_5(i)  =  Sum_Out_PID_5(i).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_Sum_5(i)  =  Sum_Out_PID_5(i).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_Sum_5(i) = sum(Int_Sum_5); % the sum of the integration of the error
    
    PID_Sum_5(i)  = Kp*Prop_Sum_5(i) + Ki*SumInt_Sum_5(i)  + Kd*Der_Sum_5(i);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                     %%%%%%%%%% Features x46, x47, and x48 %%%%%%%%%%
                             %%%%%%%%%% PID Unit (16)%%%%%%%%%%
    Prop_16(i) = x(i,46).*Error(i+1);% error of proportional term
    Der_16(i)  = x(i,47).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_16(i)  = x(i,48).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_16(i) = sum(Int_16); % the sum of the integration of the error
    
    PID_16(i)  = Kp*Prop_16(i) + Ki*SumInt_16(i)+ Kd*Der_16(i); % the three PID terms
  
                     %%%%%%%%%% Features x49, x50, and x51 %%%%%%%%%%
                             %%%%%%%%%% PID Unit (17)%%%%%%%%%%
    Prop_17(i) = x(i,49).*Error(i+1);% error of proportional term
    Der_17(i)  = x(i,50).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_17(i)  = x(i,51).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_17(i) = sum(Int_17); % the sum of the integration of the error
    
    PID_17(i)  = Kp*Prop_17(i) + Ki*SumInt_17(i)+ Kd*Der_17(i); % the three PID terms
   
                     %%%%%%%%%% Features x52, x53, and x54 %%%%%%%%%%
                             %%%%%%%%%% PID Unit (18)%%%%%%%%%%
    Prop_18(i) = x(i,52).*Error(i+1);% error of proportional term
    Der_18(i)  = x(i,53).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_18(i)  = x(i,54).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_18(i) = sum(Int_18); % the sum of the integration of the error
    
    PID_18(i)  = Kp*Prop_18(i) + Ki*SumInt_18(i)+ Kd*Der_18(i); % the three PID terms
 
                %%%%%%%%%% Sum of outputs PID Unit(16),(17),and (18)%%%%%%%%%%
    Sum_Out_PID_6(i) = PID_16(i) + PID_17(i) + PID_18(i);
    
              %%%%%%%%%% Sum of outputs entering the PID Sum Unit (6) 
    Prop_Sum_6(i) =  Sum_Out_PID_6(i).*Error(i+1);% error of proportional term
    Der_Sum_6(i)  =  Sum_Out_PID_6(i).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_Sum_6(i)  =  Sum_Out_PID_6(i).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_Sum_6(i) = sum(Int_Sum_6); % the sum of the integration of the error
    
     PID_Sum_6(i) = Kp*Prop_Sum_6(i) + Ki*SumInt_Sum_6(i) + Kd*Der_Sum_6(i);     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    PID_Total_2(i) = PID_Sum_4(i) + PID_Sum_5(i) + PID_Sum_6(i);
     
    Prop_Total_2(i) =  PID_Total_2(i).*Error(i+1);% error of proportional term
    Der_Total_2(i)  =  PID_Total_2(i).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_Total_2(i)  =  PID_Total_2(i).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_Total_2(i) = sum(Int_Total_2); % the sum of the integration of the error
    
    PID_Last_Uint_2(i) = Kp*Prop_Total_2(i) + Ki*  SumInt_Total_2(i) + Kd*Der_Total_2(i);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Feature(i) =   PID_Last_Uint_1(i) +  PID_Last_Uint_2(i);
  
    Prop_Feature_2(i) =  Feature(i).*Error(i+1);% error of proportional term
    Der_Feature_2(i)  =  Feature(i).*((((Error(i+1) - Error(i))/dt))); % derivative of the error
    Int_Feature_2(i)  =  Feature(i).*(((Error(i+1) + Error(i))*dt/2)); % integration of the error
    SumInt_Feature_2(i) = sum(Int_Feature_2); % the sum of the integration of the error
    
    PID_Final_Feature(i) = Kp * Prop_Feature_2(i) + Ki*  SumInt_Feature_2(i) + Kd*Der_Feature_2(i);
    
  
   FeedBack(i+1)=  PID_Final_Feature(i);
  
    
 Final_Output(data,:)= PID_Final_Feature(i);
 
F1(data,:)= PID_1(i);
  F2(data,:)=  PID_2(i);
    F3(data,:)=  PID_3(i);
     F4(data,:)=  PID_4(i);
      F5(data,:)=  PID_5(i);
       F6(data,:)=  PID_6(i);
         F7(data,:)=  PID_7(i);
           F8(data,:)=  PID_8(i);
             F9(data,:)= PID_9(i);
               F10(data,:)= PID_10(i);
                 F11(data,:)= PID_11(i);
                    F12(data,:)= PID_12(i);
                      F13(data,:)= PID_13(i);
                        F14(data,:)= PID_14(i);
                          F15(data,:)= PID_15(i);
                            F16(data,:)= PID_16(i);
                               F17(data,:)= PID_17(i);
                                 F18(data,:)= PID_18(i);
                                   F19(data,:)= PID_Sum_1(i);
                                      F20(data,:)= PID_Sum_2(i);
                                         F21(data,:)= PID_Sum_3(i);
                                           F22(data,:)= PID_Sum_4(i);
                                             F23(data,:)= PID_Sum_5(i);
                                               F24(data,:)= PID_Sum_6(i);
                                                 F25(data,:)= PID_Last_Uint_1(i);
                                                   F26(data,:)= PID_Last_Uint_2(i);
                                                     F27(data,:)= PID_Final_Feature(i);

end
end
end

C(chanal).Total_PIDC_Features = [F1,F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24, F25, F26, F27];

end

flattening_layer = [C(1).Total_PIDC_Features, C(2).Total_PIDC_Features, C(3).Total_PIDC_Features];
classes = cell2mat(load_images(:,1));
                                  %%% Features extraction process terminates %%%

%% Data processing Starts 
% 
AllData = [flattening_layer, classes];

t1 = AllData(1:208,:);
t2 = AllData(309:516,:); 
DataTrainValidate= [t1; t2];

number_of_data = size(DataTrainValidate,1);

t = floor(0.702* number_of_data);
v = floor (0.3*number_of_data); 

random = randperm(number_of_data);
random_train = random(1:t);
random_val = random (t+1 :t+v);

training_data   = DataTrainValidate(random_train, :);
validation_data = DataTrainValidate(random_val, :);


ts1 = AllData(209:308,:);
ts2 = AllData(517:616,:);
testing_data = [ts1; ts2];

                                      %%% Data processing terminates %%%

%%
% Training
%
Xe = training_data(:, 1:81);
D = training_data(:, 82);

W1 = (2*rand(100, 81) - 1)./ sqrt ((100^2) + (81^2));
W2 = (2*rand(100, 100) - 1);
W3 = (2*rand(100, 100) - 1)./sqrt ((100^2) + (100^2));
W4 = (2*rand( 2, 100) - 1);
rng(1);

beta  = 0.9;
alpha = 0.01;

momentum1 = zeros(size(W1));
momentum2 = zeros(size(W2));
momentum3 = zeros(size(W3));
momentum4 = zeros(size(W4));

N = length(D);
bsize = 8;
blist = 1:bsize:(N-bsize+1);
% One epoch loop
%
for epoch = 1:30
   
for batch = 1:length(blist)
 dW1 = zeros(size(W1));
 dW2 = zeros(size(W2));
 dW3 = zeros(size(W3));
 dW4 = zeros(size(W4));
 % Mini-batch loop
 %
 begin = blist(batch);
 
for k = begin:begin+bsize-1
% Forward pass = inference
 xe = Xe(k,:);
 xe = xe';
 v1 = W1*xe;
 y1 = v1;
 v2 = W2*y1;
 y2 = v2;
 v3 = W3*y2;
 y3 = v3;
 v = W4*y3;
 y = Softmax(v);
 
 % One-hot encoding
 %
 d = zeros(2, 1);
 d(sub2ind(size(d), D(k), 1)) = 1;
 % Backpropagation
 %
 e = d - y;
 delta = e;
 e3 = W4'*delta;
 delta3 = (v3 > 0).*e3;
 e2 = W3'*delta3;
 delta2 = (v2 > 0).*e2;
 e1 = W2'*delta2;
 delta1 = (v1 > 0).*e1;
 
 dW4 = dW4 + delta*y3';
 dW3 = dW3 + delta3*y2';
 dW2 = dW2 + delta2*y1';
 dW1 = dW1 + delta1*xe';
 end
 % Update weights
 %
 dW4 = dW4 / bsize;
 dW3 = dW3 / bsize;
 dW2 = dW2 / bsize;
 dW1 = dW1 / bsize;
 
 momentum4 = alpha*dW4 + beta*momentum4;
 W4 = W4 + momentum4;
 
 momentum3 = alpha*dW3 + beta*momentum3;
 W3 = W3 + momentum3;
 
 momentum2 = alpha*dW2 + beta*momentum2;
 W2 = W2 + momentum2;
 
 momentum1 = alpha*dW1 + beta*momentum1;
 W1 = W1 + momentum1;

% Training performance at each iteration
%
Xp = training_data(:, 1:81);
Dp= training_data(:, 82);
accp = 0;
Np = length(Dp);

for k0 = 1:Np
 xp = Xp(k0,:);
 xp = xp';
 vp1 = W1*xp;
 yp1 = vp1;
 vp2 = W2*yp1;
 yp2 = vp2;
 vp3 = W3*yp2;
 yp3 = vp3;
 vp = W4*yp3;
 yp= Softmax(vp);
 
% Loss calculation of Training process
Ground_truthp = Dp(k0,1);

if (Ground_truthp == 1)
    Predicted_valuep= yp(1,1);
    lossp(k0) = -log(Predicted_valuep);
else 
    Predicted_valuep= yp(2,1);
    lossp(k0) = -log(Predicted_valuep);
end 

% Accuracy calculation of Training process
%
 [~, i] = max(yp);
 predictionp(k0,:) = i;
 
 if (i == Dp(k0))    
 accp= accp + 1;
 end
 
end

Accp(epoch).accp(batch) = (accp / Np).*100;
Lossp(epoch).lossp(batch) = sum(lossp)/Np; 

end

% Valdition 
%

Xv = validation_data(:, 1:81);
Dv = validation_data(:, 82);
accv = 0;
Nv = length(Dv);

for k1 = 1:Nv
 xv = Xv(k1,:);
 xv = xv';
 vv1 = W1*xv;
 yv1 = vv1;
 vv2 = W2*yv1;
 yv2 = vv2;
 vv3 = W3*yv2;
 yv3 = vv3;
 vv = W4*yv3;
 yv= Softmax(vv);
 
% Loss calculation of validation process
Ground_truthv = Dv(k1,1);

if (Ground_truthv == 1)
    Predicted_valuev= yv(1,1);
    lossv(k1) = -log(Predicted_valuev);
else 
    Predicted_valuev= yv(2,1);
    lossv(k1) = -log(Predicted_valuev);
end 

% Accuracy calculation of validation process
%

 [~, ii] = max(yv);
 predictionv(k1,:) = ii;
 
 if (ii == Dv(k1))    
 accv= accv + 1;
 end
 
end

Accv.accv(epoch) = (accv / Nv).*100;
Lossv(epoch) = sum(lossv)/Nv;

end

%%
VectorizationA =vertcat( Accp.accp );
Training_Accuracy =  reshape(VectorizationA.',1,[]);
VectorizationL = vertcat(Lossp.lossp);
Training_Loss =reshape(VectorizationL.',1,[]);

Validation_Accuracy= Accv.accv;
Validation_Loss = Lossv;


figure(1);
plot(Training_Accuracy, 'r');
xlabel('Epoch');
ylabel('Training Accuracy');

figure(2);
plot(Training_Loss, 'r');
xlabel('Epoch');
ylabel('Training Loss');


figure(3);
plot(Validation_Accuracy, 'r');
xlabel('Epoch');
ylabel('Validation Accuracy');

figure(4);
plot(Validation_Loss, 'r');
xlabel('Epoch');
ylabel('Validation Loss');

toc
%%
% Test
% 
tic
Xt = testing_data(:, 1:81);
Dt = testing_data(:, 82);
acct = 0;
Nt = length(Dt);

for k2 = 1:Nt
 xt = Xt(k2,:);
 xt = xt';
 vt1 = W1*xt;
 yt1 = vt1;
 vt2 = W2*yt1;
 yt2 = vt2;
 vt3 = W3*yt2;
 yt3 = vt3;
 vt = W4*yt3;
 yt = Softmax(vt);
 
 [~, iii] = max(yt);
 prediction(k2,:) = iii;
 if iii== Dt(k2)
 acct = acct + 1;
 end
end
acct = (acct / Nt).*100;
fprintf('Accuracy is %f\n', acct)
figure
confusionchart(Dt,prediction);
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                        %%% The End of Algorithm %%%
                                    %%% Refat Mohammed Abdullah Eshaq %%%
                               %%% <https://orcid.org/0000-0002-6448-4054> %%%
