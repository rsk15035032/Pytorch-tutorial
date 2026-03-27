### 📘 This you find the end-to-end implementation of Pytorch in the DeepLearing. The topics are cover below:
1. [Initializing Tensors](tensor_1.py)

2. [Tensor Math & Comparison Operations](tensor_2.py)

3. [Tensor Indexing & Slicing](tensor_3.py)

4. [Tensor Reshaping & Manipulation](tensor_4.py)

5. [Build the first Sample Nural Network.](nuralNetwork.py)

6. [Build basics CNN Model.](CNN/cnn.py)

7. [Build basic RNN Model.](RNN/rnn.py)

8. [Using Bidirectional LSTM.](RNN/BidirectionalLSTM.py)

9. [Save  and load model in Checkpoints.](CNN/loadSave.py)

10. [Fine tuning and Transfer Learning of pretrained VGG16 model.](TransferLearning+FineTuning/TransferLearningandFineTuning.py)

11. [Import data and train on pretrained Vgg16 of googlenet.](Build_Custom_Datasets/ImportData.py)

12. [Build the custom dataset loading for image captioning on flickr8k.](Build_custom_textDatasets/text_data.py)

13. [Data Augmentation of images using the torchvision.](Buid_Custom_Datasets/augmentation.py)

14. [Apply albumentation on the image.](Albumentation)

15. [Method dealing with imbalaced datasets using oversampling and class weighting.](Imbalanced_classes/main.py)

16. [Run the Tensorboard to evaluate the model on MNIST dataset.](Tensorboard/main.py)

17. [Implementation of Lenet architech from scratch.](cnn_architectures/lenet/lenet5_pytorch.py)

18. [Implementation of VGG16 as well as generalize VGG architecture from scratch.](cnn_architectures/VGG/vgg_pytorch.py)

19. [Implementation of GoogLeNet/InceptionNet from scratch.](cnn_architectures/InceptionNet/InceptionNet_pytorch.py)

20. [Implementation of ResNet from scratch.](cnn_architectures/ResNet/ResNet_pytorch.py)

21. [Implementation of EfficientNet from scratch.](cnn_architectures/EfficientNet/efficientNet_pytorch.py)

22. [Implementation of Image Captioning from scratch.](Image_Captioning)

23. [Implementation of Neral style transfer using Pytorch.](nuralStyle)

24. [Implementation of Simple GAN from scratch.](GANs/SimpleGAN/simpleGAN.py)

25. [Implementation of DCGAN from scratch.](GANs/DCGAN)

26.[Implementation of Wasserstein Generative Adversarial Networks (WGANs) from scratch.](GANs/WGAN/)

27. [Implementation of Wasserstein Generative Adversarial Networks (WGANs) with gradient penality from scratch.](GANs/WGAN-GP/)

28. [Implementation of Conditional GAN on WGAN with gradient penality from scratch.](GANs/ConditionalGAN/)

29. [Implementation of Patch GAN on UNET for Pix2Pix on Map dataset from scratch.](GANs/Pix2Pix/)

30. [Implementation of CycleGAN to horse to zebra datasets from scratch.](GANs/CycleGAN/)

31. [Implementation of ProGAN on WGAN-gradient penality from scratch.](GANs/ProGAN/)

32. [Implementation of SRGAN from low to high image resolution from scratch.](GANs/SRGAN/)

33.  [Implementation of ESRGAN from low to high image resolution from scratch.](GANs/ESRGAN/)

34. [Implementation of Baby name generator using RNN-LSTM from scratch.](TextGenerator/nameGenerator.py)

35. [This project demonstrates a complete LSTM text classification pipeline in PyTorch using TorchText for dataset processing, batching, training, and model persistence.](torchText/part1/part1.py)

36. [TorchText + SpaCy implementation for loading and preprocessing the Multi30k German–English dataset for sequence models like LSTM or Seq2Seq.](torchText/part2/part2.py)

37. [TorchText-based English–German translation data pipeline with spaCy tokenization, vocabulary construction, and CPU/GPU-ready batch iterators.](torchText/part3/part3.py)

38. [Implementation of a Seq2Seq (LSTM-based) neural machine translation model using PyTorch and TorchText for German-to-English translation with training, evaluation (BLEU score), and inference utilities.](seq2seq/machineTranslation)

39. [Implementation of an attention-based Seq2Seq neural machine translation model using PyTorch with training, inference, BLEU evaluation, and checkpointing](seq2seq/machineTranslationWithAttention/)

40. [Built a Transformer model from scratch using PyTorch with multi-head self-attention and encoder–decoder architecture.](Transformer/transformer.py)

41. [Implementation of a Transformer-based Seq2Seq model for German-to-English translation using PyTorch, trained on the Multi30k dataset with GPU/CPU support, tqdm training progress, TensorBoard logging, and BLEU score evaluation.](transformer_seq2seq/transformer_machinetranslation.py)

42. [U-Net based image segmentation project implemented in PyTorch with custom dataset, training pipeline, and prediction visualization.](UNET_imageSegmentation)

43. [A clean PyTorch implementation of Intersection over Union (IoU) to measure the overlap between predicted and ground-truth bounding boxes for object detection models.](Object_detection/metrics/IoU.py)

44. [Removes overlapping bounding boxes using IoU threshold and confidence score to keep only the best predictions.](Object_detection/metrics/nms.py)

45. [Computes Mean Average Precision (mAP) by matching predicted and ground-truth bounding boxes using IoU and integrating the precision–recall curve.](Object_detection/metrics/mAP.py)

46. [End-to-end YOLOv1 object detection implementation in PyTorch including model, loss, dataset, and training pipeline.](Object_detection/YOLOv1)

47. [Optimized YOLOv3 (PyTorch) project with multi-scale detection, custom loss, and Pascal VOC training](Object_detection/YOLOv3)

48. [A simple and optimized Convolutional Neural Network (CNN) implemented in PyTorch for MNIST digit classification with GPU and mixed-precision (FP16) support.](QuickTips/fp16.py)

49. [Beginner-friendly PyTorch CNN project demonstrating a clean training loop with real-time progress tracking using tqdm.](QuickTips/progress_bar.py)

50. [Utility function to make PyTorch experiments fully reproducible by fixing random seeds across Python, NumPy, CPU, and GPU.](QuickTips/set_seeds.py)

51. [Script to calculate mean and standard deviation of image datasets (CIFAR-10) for deep learning normalization.](QuickTips/std_mean.py)