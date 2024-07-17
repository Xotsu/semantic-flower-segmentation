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
I = readimage(imds,609);
figure
imshow(I)
C = readimage(pxds,609);
C(5,5);

% show overlaid groundtruth labels as an example
B = labeloverlay(I,C);
figure
imshow(B)

% pair images with labels & split data for training, validation and testing
% training: 60%, validation: 20%, testing: 20%
% Function altered from https://uk.mathworks.com/help/vision/ug/semantic-segmentation-using-deep-learning.html
[train_data, validate_data, imds_test, pxds_test] = partitionData(imds, pxds, 0.6, 0.2);


% load pretrained network
input_size = [256, 256, 3];

% net = segnetLayers(input_size, 2, "vgg16");

% net = unetLayers(input_size, 2);

net = deeplabv3plusLayers(input_size, 2, "resnet50");

% training options
opts = trainingOptions("adam", ...
    "InitialLearnRate",1e-4, ...
    "MaxEpochs",40, ...
    "ValidationData", validate_data, ...
    "ExecutionEnvironment","gpu", ...
    "Plots", "training-progress", ...
    "OutputNetwork","best-validation-loss", ...
    "LearnRateSchedule","piecewise", ...
    "Verbose", true, ...
    "MiniBatchSize",6);

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

% train the network
net = trainNetwork(train_data,net,opts);

% save the network

save('segmentexistnet.mat', 'net')

% comment above and uncomment below to load and test model
% model = load("segmentexistnet.mat");
% net = model.net;

% do segmentation, save output images to disk (needs "out" folder)
pxds_results = semanticseg(imds_test,net);

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
overlay_out = labeloverlay(readimage(imds_test,30),readimage(pxds_results,30));
figure
imshow(overlay_out);
title("overlay out")

overlay_out = labeloverlay(readimage(imds_test,31),readimage(pxds_results,31));
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

