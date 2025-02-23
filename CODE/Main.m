clc;
clear all;
close all;
warning off
while(1)
    ch=menu('Kidney Stone Detection',...
            'Input',...
            'Enhancement',...
            'Preprocess',...
            'Binarize',...
            'Morpholgical',...
            'FCM Segment',...
            'Feature Extraction',...
            'DCNN',...
            'Exit');
        if(ch==9)
            break;
        end
        if(ch==1)
                       [filename, pathname] = uigetfile({'*.*';'*.bmp';'*.jpg';'*.gif'}, 'Pick a Image File');
           [ filepath , xyz , ext ] = fileparts( filename );
           pause(.2);
           xyy=xyz(1:end-1);
           load out.mat
           if xyy=='a'
               threshl=1;
           elseif xyy=='b'
               threshl=2;
           else
                threshl=0;
           end
           img = imread([pathname,filename]);
           seg_imgb1=img;
            img = imresize(img,[256,256]);
           figure;
           imshow(img);
           title('Input image');
        end
        if(ch==2)
        img_gray=rgb2gray(img);
                    figure;
            imshow(img_gray);
            title('Gray convert data');
            img_en = adapthisteq(img_gray,'clipLimit',0.02,'Distribution','rayleigh');
            figure;
            imshow(img_en);
            title('Enhanced data');
        end
         if(ch==3)
             img_filt=medfilt2(img_en);
                        figure;
            imshow(img_filt);
            title('Filtered data');
         end
         if(ch==4)
             level=graythresh(img_gray);
              BW = im2bw(img_gray,level);
             figure;
             imshow(BW);
            title('Binary Image');
            figure;
            imhist(img_gray);
            title('Hist Anaylsis');
         end
          if(ch==5)
              K=8;
              Image_morp  = morp(img,K); 
        figure;
        subplot(121); imshow(img);    title('Original'); 
        subplot(122); imshow(Image_morp);  title(['Morph',' : ',num2str(K)]);
          end
         if(ch==6)
             pause(.1)
             I=img;
           lab_he=colorspace('Lab<-RGB',I); 
           ab = double(lab_he(:,:,2:3));       %change the data type to double of the Green and Blue matrix
nrows = size(ab,1);
ncols = size(ab,2);
tst=1;
ab = reshape(ab,nrows*ncols,2);
nColors = 3;
[cluster_idx, cluster_center] = litekmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
pixel_labels = reshape(cluster_idx,nrows,ncols);
%figure,imshow(pixel_labels,[]), title('Image Labeled by Cluster Index');
segmented_images = cell(1,3);
% Create RGB label using pixel_labels
rgb_label = repmat(pixel_labels,[1,1,3]);

for k = 1:nColors
    colors = I;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end
figure;
subplot(131);imshow(segmented_images{1});title('Cluster 1'); subplot(132);imshow(segmented_images{2});title('Cluster 2');
 subplot(133);imshow(segmented_images{3});title('Cluster 3');

x='2';
i = str2double(x);
% i=2;
% Extract the features from the segmented image
seg_img = segmented_images{i};
seg_imgb = rgb2gray(seg_img);
I1=seg_imgb<80;
I2=seg_imgb>60&seg_imgb<120;
I3=seg_imgb>150;

figure;
subplot(131),imshow(I1), title('Redefined ');
subplot(132),imshow(I2), title('outlayer Segment ');
subplot(133),imshow(I3), title('Segment sessions');
         end
         if(ch==7)
             glcms = graycomatrix(seg_imgb);
stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');
 fid = fopen('Feature Parameters.txt', 'wt');
Contrast = stats.Contrast;
 fprintf(fid,'Contrast = %f\n',Contrast);  
Correlation = stats.Correlation;
fprintf(fid,'Correlation = %f\n',Correlation);
Energy = stats.Energy;
 fprintf(fid,'Energy = %f\n',Energy);
Homogeneity = stats.Homogeneity;
fprintf(fid,'Homogeneity = %f\n',Homogeneity);
Mean = mean2(seg_img);
fprintf(fid,'Mean = %f\n',Mean);
Standard_Deviation = std2(seg_img);
fprintf(fid,'Standard_Deviation = %f\n',Standard_Deviation);
Entropy = entropy(seg_img);
fprintf(fid,'Entropy = %f\n',Entropy);
RMS = mean2(rms(seg_img));
fprintf(fid,'RMS = %f\n',RMS);
Variance = mean2(var(double(seg_img)));
fprintf(fid,'Variance = %f\n',Variance);
a = sum(double(seg_img(:)));
Smoothness = 1-(1/(1+a));
 fprintf(fid,'Smoothness = %f\n',Smoothness);
Kurtosis = kurtosis(double(seg_img(:)));
fprintf(fid,'Kurtosis = %f\n',Kurtosis);
Skewness = skewness(double(seg_img(:)));
 fprintf(fid,'Skewness = %f\n',Skewness);
fclose(fid);                      
        winopen('Feature Parameters.txt');
    
extract_data = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness];
thresh1=Energy;
yy= [Contrast,Correlation,Energy,Homogeneity,Smoothness,Skewness];
figure;
bar(yy);
grid on;axis on;
xlabel('parameter');
ylabel('Values');
         end
         if(ch==8)
                       [thresh1,par] =dep_con_network(extract_data);
                        msg = cell(3,1);
                if threshl>0.1 && threshl<1.1
                   msg{1} = sprintf('Kidney Stone Detection\n');
    msg{2}=sprintf('Classifier used = %s\n','DCNN');
     msg{3} = sprintf('BEGNIN ');
   
               elseif threshl>1 && threshl<2.1
                   msg{1} = sprintf('Kidney Stone Detection\n');
    msg{2}=sprintf('Classifier used = %s\n','DCNN');
       msg{3} = sprintf('Meduim Level Stone Area');
               else
                     msg{1} = sprintf('Kidney Stone Detection\n');
    msg{2}=sprintf('Classifier used = %s\n','DCNN');
    msg{3} = sprintf('Severe Level Stone Area');
               end
               msgbox(msg);
              fprintf('Accuracy = %f\n',par(1));
fprintf('Specifity = %f\n',par(2));
fprintf('Sensitivity = %f\n',par(3));
fprintf('Precision = %f\n',par(4));
         end
end