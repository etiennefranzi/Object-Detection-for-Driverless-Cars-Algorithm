# Object Detection for driverless car video footage



## Project Files

- [Final Product](https://drive.google.com/file/d/1EiD5pbMaVzlxln0d-fqmlaiCpwSfuWlh/view?usp=sharing)

	[![Video preview](https://drive.google.com/thumbnail?id=1EiD5pbMaVzlxln0d-fqmlaiCpwSfuWlh&sz=w400)](https://drive.google.com/file/d/1EiD5pbMaVzlxln0d-fqmlaiCpwSfuWlh/view?usp=sharing)

- [Project Presentation](https://docs.google.com/presentation/d/1HlO5TBtLLhUnKgaEPrjG1ameoiroPUlvjU_UTfX41Mk/edit?usp=sharing)

	[![Presentation preview](https://drive.google.com/thumbnail?id=1HlO5TBtLLhUnKgaEPrjG1ameoiroPUlvjU_UTfX41Mk&sz=w400)](https://docs.google.com/presentation/d/1HlO5TBtLLhUnKgaEPrjG1ameoiroPUlvjU_UTfX41Mk/edit?usp=sharing)


Vehicle Detection Pipeline for Autonomous Driving Footage

Created Python algorithms to pinpoint and classify objects (cars, trucks, background) within video frames captured from a driving car. The final product could process 10 seconds of car footage in approximately 60 seconds on Google Colab.

Baseline classification: Started with a basic perceptron trained on a filtered vehicle subset of CIFAR-10. It got to ~95% training accuracy but only ~72% on test data, a good early lesson in how much fully-connected networks struggle with raw pixel input.
CNN from scratch: Built a CNN (Conv2D → MaxPooling → Dense layers) to actually exploit the spatial structure of the images, which improved accuracy noticeably over the perceptron.
Sliding-window localisation: Implemented a sliding-window approach, cropping and scanning fixed-size windows across each image and classifying each one individually. It worked, but was slow and pretty naive.
Transfer learning: Swapped in CNNs pretrained on ImageNet (VGG16, VGG19, ResNet50, DenseNet121), keeping their convolutional base and training custom dense layers on top. VGG16 gave the best results, reaching about 95% validation accuracy.
YOLO / DarkNet for localisation: Moved away from sliding-window in favor of YOLOv3's DarkNet architecture, which predicts bounding boxes and class probabilities across the whole image in one pass, at three different scales. Implemented the post-processing pipeline myself — decoding raw network output into bounding boxes, filtering by an objectness threshold, and applying Non-Maximal Suppression to clean up overlapping detections.
Video detection: Extended the image-level detection to work frame-by-frame on video using OpenCV, converting between OpenCV and PIL formats to run detection and rebuild the output video.

Takeaway: for classification, transfer learning from established CNNs (VGG16, DarkNet) beat anything trained from scratch. For localisation, YOLO's single-pass, grid-based approach was both faster and more accurate than sliding windows.

Tech stack: Python, TensorFlow/Keras, OpenCV, NumPy, Google Colab (GPU runtime)

What I learned: This project made the tradeoffs in object detection really concrete — accuracy vs. speed, training from scratch vs. transfer learning, and why naive approaches like sliding windows don't scale once you need real-time or near-real-time performance. It also gave me a much better feel for how CNN architectures like VGG and DarkNet actually process images through their layers, rather than just knowing them as black boxes.
