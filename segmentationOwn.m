close all;
clear;

rng(0);
% load all label files
label_files = dir(fullfile("labels_256/*.png"));

% load all image files with label file names
label_names = {label_files.name};
image_names = strrep(label_names,".png",".jpg");

% get paths for images and labels
label_paths = fullfile("labels_256/",label_names);
image_paths = fullfile("images_256/", image_names);

% load data

imds = imageDatastore(image_paths);

class_names = ["flower", "background"];
% column vector cell array mapping multiple pixel label ids to one class
pixel_label_ids = {1, [2; 3; 4]};
pxds = pixelLabelDatastore(label_paths, class_names, pixel_label_ids);

% used for testing label loading and noticing flower label noise
% show first input image
I = readimage(imds,1);
figure
imshow(I)
C = readimage(pxds,1);
C(5,5);

% show overlaid groundtruth labels as an example
B = labeloverlay(I,C);
figure
imshow(B)

% pair images with labels & split data for training, validation and testing
% training: 60%, validation: 20%, testing: 20%
% Function altered from https://uk.mathworks.com/help/vision/ug/semantic-segmentation-using-deep-learning.html
[train_data, validate_data, imds_test, pxds_test] = partitionData(imds, pxds, 0.6, 0.2);

% class weight balancing
% builds a table of class frequencies with & for the entire dataset 
% (more data = better estimation)
tbl = countEachLabel(pxds)
pixel_count = sum(tbl.PixelCount);
frequency = tbl.PixelCount / pixel_count;
class_weights = 1./frequency

% input layer
input_size = [256, 256, 3];
img_layer = imageInputLayer(input_size);

% downsampling
batch_norm = batchNormalizationLayer();
relu = reluLayer();

pool_size = 2;

downsampling_layers = [
    convolution2dLayer(5,256,"Padding",2);
    batch_norm
    relu
    maxPooling2dLayer(5,"Stride",2);
    convolution2dLayer(3,256,"Padding",2);
    relu
    maxPooling2dLayer(5,"Stride",2);
    convolution2dLayer(3,128,"Padding",1);
    relu
    maxPooling2dLayer(2,"Stride",2);
    convolution2dLayer(3,64,"Padding",1);
    relu
    maxPooling2dLayer(2,"Stride",2);
    dropoutLayer(0.4);
    ]

% upsampling

upsampling_layers = [
    transposedConv2dLayer(4,64,"Stride",2);
    relu
    transposedConv2dLayer(4,128,"Stride",2,"Cropping",1);
    relu
    transposedConv2dLayer(4,128,"Stride",2,"Cropping",1);
    relu
    transposedConv2dLayer(4,256,"Stride",2,"Cropping",1);
    relu
    ]

% output
num_classes = 2;
conv1x1 = convolution2dLayer(1,num_classes);

final_layers = [
    conv1x1
    softmaxLayer()
    % pixel classifier balanced on class weights
    pixelClassificationLayer('Classes',tbl.Name,'ClassWeights',class_weights)
    ]

% net layers
net = [
    img_layer    
    downsampling_layers
    upsampling_layers
    final_layers
    ]

% training options
opts = trainingOptions("adam", ...
    "InitialLearnRate",1e-4, ...
    "MaxEpochs",60, ...
    "ValidationData", validate_data, ...
    "ExecutionEnvironment","gpu", ...
    "Plots", "training-progress", ...
    "OutputNetwork","best-validation-loss", ...
    "ValidationFrequency", 25, ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropPeriod",25, ...
    "Verbose", true, ...
    "MiniBatchSize",8);


% displays a batch from augmented data
augmented_batch = read(train_data);
% display x augmented images with labels depending on batch size
for i = 1:min(1, size(augmented_batch, 1))
    % overlays labels on the image
    overlay_out = labeloverlay(augmented_batch{i, 1}, augmented_batch{i, 2});
    figure;
    imshow(overlay_out);
    title('Augmented Image');
end
% 
% train the network
net = trainNetwork(train_data,net,opts);

% save the network
save('segmentownnet.mat', 'net')

% comment above and uncomment below to load and test model
% model = load("segmentownnet.mat");
% net = model.net;

%deepNetworkDesigner(net)

% do segmentation, save output images to disk (needs "out" folder)
pxds_results = semanticseg(imds_test,net, 'MiniBatchSize', 8);

% loop through first 6 images of test data (shuffled when partitioning)
% display the predicted labels in a grid
figure;
tiledlayout(2,3);
for i = 1:6
    overlay_out = labeloverlay(readimage(imds_test,60+i), readimage(pxds_results,60+i));

    nexttile;
    imshow(overlay_out);
    % title(sprintf("Overlay Out %d", i));
end


%show a couple of output images, overlaid
overlay_out = labeloverlay(readimage(imds_test,35),readimage(pxds_results,35));
figure
imshow(overlay_out);
title("overlay out")

overlay_out = labeloverlay(readimage(imds_test,45),readimage(pxds_results,45));
figure
imshow(overlay_out);
title("overlay out 2")

% evaluation
metrics = evaluateSemanticSegmentation(pxds_results,pxds_test)

figure
cm = confusionchart(metrics.ConfusionMatrix.Variables, ...
  class_names, Normalization='row-normalized');

cm.Title = 'Normalized Confusion Matrix (%)';

image_IoU = metrics.ImageMetrics.MeanIoU;
figure
histogram(image_IoU)
title('Image Mean IoU')




% function altered from https://uk.mathworks.com/help/vision/ug/semantic-segmentation-using-deep-learning.html
function [train_data, validate_data, imds_test, pxds_test] = partitionData(imds,pxds, training_perc, validation_perc)

% shuffles the datastore indices
num_files = numpartitions(imds);
shuffled_indices = randperm(num_files);

num_train = round(training_perc * num_files);
training_idx = shuffled_indices(1:num_train);
num_validate = round(validation_perc * num_files);
validate_idx = shuffled_indices(num_train+1:num_train+num_validate);

% whatever's left used for testing
test_idx = shuffled_indices(num_train+num_validate+1:end);

% creates new datastores from the subset indexes
imds_train = subset(imds,training_idx);
imds_validate = subset(imds,validate_idx);
imds_test = subset(imds,test_idx);
pxds_train = subset(pxds,training_idx);
pxds_validate = subset(pxds,validate_idx);
pxds_test = subset(pxds,test_idx);

% combine the images and labels

train_data = combine(imds_train, pxds_train);
validate_data = combine(imds_validate, pxds_validate);

% training data augmentation
train_data = transform(train_data, @(data)augmentImageAndLabel(data));

end

function data = augmentImageAndLabel(data)

for i = 1:size(data,1)
    % REMOVED rotation between -45 and 45 degrees
    % angle = [30, 30];
    % , ...
    %     Rotation=angle

    % scaling between 1 and 1.2
    scale = 1 + 0.2 * rand();
    % applies effects + 50% chance of reflection
    tform = randomAffine2d(...
        XReflection=true, ...
        Scale=[scale, scale]);
    % outputs view for warped images
    rout = affineOutputView(size(data{i,1}), tform, BoundsStyle="centerOutput");

    % warps images & labels using transformation
    data{i,1} = imwarp(data{i,1}, tform, OutputView=rout);
    data{i,2} = imwarp(data{i,2}, tform, OutputView=rout);

    % % converts to hue, saturation & value
    % hsv_image = rgb2hsv(data{i,1});
    % % saturation between 0.8 and 1.2
    % hsv_image(:,:,2) = hsv_image(:,:,2) * (0.9 + 0.2* rand());
    % % hue adjustment between -0.05 and 0.05
    % hsv_image(:,:,1) = hsv_image(:,:,1) + (0.1 * rand() - 0.05);
    % % converts back to rgb
    % data{i,1} = hsv2rgb(hsv_image);
    
    % applies nd and median filtering to image and labels to reduce noise
    % data{i,1} = imfilter(data{i,1}, ones(5,5)/25);
    
    % need to convert categorical labels to numerical for filter
    numeric_labels = uint8(data{i,2});
    % apply median filter to numeric labels
    filtered_labels = medfilt2(numeric_labels, [5, 5], "symmetric");
    % rebuilds the labels with filters based on unique numerical value &
    % matching categories
    original_categories = categories(data{i,2});
    original_values = 1:numel(original_categories);
    data{i,2} = categorical(filtered_labels, original_values, original_categories);

end
end

