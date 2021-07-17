clear all;
clc;

% Load both images
RGB1 = imread('im1.jpg');
RGB2 = imread('im2.jpg');

% convert both images to double and to gray scale
img1 = im2double(rgb2gray(RGB1));
img2 = im2double(rgb2gray(RGB2));

% initialize
harris_sigma = 2;
harris_thresh = 0.05;
harris_radius = 2;

% Detect feature points using harries corner detector in both images
[~,y1,x1] = harris(img1,harris_sigma,harris_thresh,harris_radius,0);
[~,y2,x2] = harris(img2,harris_sigma,harris_thresh,harris_radius,0);

sift_radius = 5;

% Compute descriptors around every keypoint in both images
descriptors1 = find_sift(img1,[x1,y1, sift_radius*ones(length(x1),1)],1.5);
descriptors2 = find_sift(img2,[x2,y2, sift_radius*ones(length(x2),1)],1.5);

num_putative_matches = 200;
% Compute matches between every descriptor in one image and every descriptor in the otherimage. Then select the best matches.
[matches1,matches2] = select_matches(descriptors1,descriptors2,num_putative_matches);
XY1 = [x1(matches1),y1(matches1)];
XY2 = [x2(matches2),y2(matches2)];

% run ransac
num_ransac_iter = 5000;
[H,num_inliers,residual] = ransac(XY1,XY2,num_ransac_iter,@homography1,@homography2);

% Warp one image onto the other using the estimated transformation
[~, xdata, ydata] = imtransform(RGB1,maketform('projective',H'));

xdata_out=[min(1,xdata(1)) max(size(RGB2,2), xdata(2))];
ydata_out=[min(1,ydata(1)) max(size(RGB2,1), ydata(2))];

result1 = imtransform(RGB1, maketform('projective',H'), 'XData',xdata_out,'YData',ydata_out);
result2 = imtransform(RGB2, maketform('affine',eye(3)), 'XData',xdata_out,'YData',ydata_out);

% composite by simply averaging the pixel values where the two images overlap
panorama = result1 + result2;
overlap = (result1 > 0.0) & (result2 > 0.0);
result_avg = (result1/2 + result2/2);

panorama(overlap) = result_avg(overlap);

% show average and panorama in grayscale
figure
imshow(rgb2gray(result_avg))
figure
imshow(rgb2gray(panorama))

