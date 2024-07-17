# Semantic Flower Segmentation: Pretrained and Customised

## Full Paper

[View](./report.pdf)

## Abstract

Semantic segmentation research plays an important role in
image analysis, notably contributing to advancements in
medical, botanical fields and agricultural automation. This
study evaluates the performance of two deep learning
models on a given Oxford flower dataset. The widely
adopted DeepLabV3+ model using ResNet50 backbone and
an individually developed encoder-decoder architecture
inspired by the SegNet architecture. An ablation study
performed during the custom model’s development
demonstrates significant and relevant architectural features
for flower semantic segmentation. The pretrained
DeepLabV3+ model, fine-tuned with the Adam optimizer
achieved a mean accuracy of 99.20% and a mean
intersection over union (IoU) of 98.63%, achieving superior
performance. In comparison, the custom model reached a
mean accuracy of 97.37% and a mean IoU of 94.02%. These
results highlight the effectiveness of tailored architectures,
approaching the performance of superior and more complex
state-of-the-art models for this dataset.

## Example Outputs

### Existing deeplabv3+ resnet50 6 image output

![Existing Output](./existing%20deeplabv3+%20resnet50%206%20image%20output.png)

### Proposed custom network 6 image output

![Proposed Output](./proposed%20custom%20network%206%20image%20output.png)