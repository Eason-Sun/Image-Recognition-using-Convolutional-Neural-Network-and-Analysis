# Image-Recognition-using-Convolutional-Neural-Network-and-Analysis

## Objective:
This project constructs a CNN model using Keras to perform image recognition tasks. An thorough analysis is then conducted for fine-turning  hyperparameters and comparision between various techniques.

## Dataset:
The CIFAR-10 dataset: cs.toronto.edu/~kriz/cifar.html

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 

Here are the classes in the dataset, as well as 10 random images from each:

![Capture](https://user-images.githubusercontent.com/29167705/63813995-651a1500-c8fd-11e9-98c6-0df86eb83eb6.JPG)

## Training Process:
The CNN is trained for 20 epochs in each following comparision. Although 20 epochs is not sufficient to reach convergence, it is sufficient for the analysis.

## Accuracy Comparision 

### 1) CNN vs Dense NN with various number of hidden layers: 

In this case, simple dense neural networks is used with 0, 1, 2, 3 and 4 hidden layers of 512 rectified linear units with a dropout rate of 0.5.

![Capture](https://user-images.githubusercontent.com/29167705/63814337-85969f00-c8fe-11e9-9fab-1c5700577712.JPG)

Overall, training accuracy and testing accuracy follow the similar trend, but testing accuracy experiences noticeably fluctuation as the number of epochs increase.

We could clearly observe that convolutional neural networks completely outperform any fully connected neural networks in terms of accuracy. When it comes to image data, CNN has huge advantages over fully connected NN, for example, equivariance and invariance to small translations of the object.

When the number of hidden layers of fully connected NN varies, the correlation between accuracy and the number of hidden layers was not found. In general, by adding more hidden layers with fixed width, NN could gain more expressiveness. The results showed that the number of hidden layers is dependent on the specific problems, therefore, it is regarded as a hyperparameter and required to finely tuned by cross validation.

### 2) CNN with sigmoid units vs reLU:

![Capture](https://user-images.githubusercontent.com/29167705/63814887-7e709080-c900-11e9-9a54-cf288b0411ba.JPG)

Overall, training accuracy and testing accuracy follow the similar trend, but testing accuracy experiences noticeably fluctuation as the number of epochs increase.

As shown in above figures, CNN with Rectified Linear Unit activation functions performs significantly well against the same network architecture with Sigmoid activation functions. During the first 4 epochs, accuracies of CNN with Sigmoid remain constant in both training testing phases, which might be due to the gradient vanishing problem which usually happens when backpropagating deep neural networks. (The derivative of the sigmoid function is less than one, by applying the chain rule, weights get barely updated.)

### 3) CNN with and without drop out as well as with and without data augmentation:

![Capture](https://user-images.githubusercontent.com/29167705/63814992-e0c99100-c900-11e9-9a6e-c0b5e05af852.JPG)

The accuracy results from training and testing share some similarities but also show different patterns.

Similarities: In general, with data augmentation, high accuracies (70% ~ 75%) were observed from very beginning (< 5 epochs). We may argue that, when the data is limited, data augmentation could add variance to the data so that the trained models can be resistant to small changes and generalized well. Another similarity is that CNN with both dropout and data augmentation has the worst overall performance. It is worth noting that, when applying both techniques, we introduce more variance/uncertainty, which explains the fact that the its accuracy curve fluctuate the most in the testing phase.

Differences: In training phase, there is no doubt that the model without both dropout and data augmentation is the best performer. However, testing phase reveals that the models with either dropout or data augmentation yield the better results. We believe that, without both techniques, the model is perfectly tuned to classify the training data, but it could not generalize that well on testing data, which leads to the overfitting problem.

### 4) CNN with three different optimizers: RMSprop, Adagrad and Adam:

![Capture](https://user-images.githubusercontent.com/29167705/63815092-41f16480-c901-11e9-8de5-91bdd6d96c83.JPG)

By choosing different optimizers, the performance of the same network architecture could differ greatly. As we can see from the above comparisons from both training and testing stages, compared to Adagrad, RMSprop is substantially better since it tends to forget previous gradients, so it eventually overcomes the drawback from Adagrad whose learning rate decays too quickly. By inducing momentum, Adam outperforms RMSprop in their training phase, but the advantage gets reduced when it comes to testing. Nevertheless, we could still see that the momentum accelerates the convergence, which makes training process a lot quicker.

### 5) CNN with larger filter but shallower layers:
Baseline CNN: Two stack of (CONV2D, Activation, CONV2D, Activation) layers 3x3 filters.   
Compared with: Two stack of (CONV2D, Activation) layers with 5x5 filters.

For this particular comparision, CNN is trained for 100 epochs.

![Capture](https://user-images.githubusercontent.com/29167705/63815374-5aae4a00-c902-11e9-9075-874a52ebf8ad.JPG)

Overall, training accuracy and testing accuracy follow the similar trend, but testing accuracy experiences noticeably fluctuation as the number of epochs increase.

We notice that the model with deeper layers and smaller filters consistently outperforms the one with shallower layers but larger filters. The relatively smaller filters could focus on fine detail features, whereas larger filters describe coarser shapes. In addition, the extra layer adds more abstraction. From this point of view, we may argue that the former might yield higher accuracy.

## Discussion on Max Pooling Layers:

CNN without max pooling layer is not invariant to the translation of the foreground object. Basically, the convolutional operation is equivariant to translation, which means that the translation of the object would result in translation of the kernel output if kernel stride = 1. For example, kernel has size 5, and it is defined to return 1 if it finds pattern 123, if input 00123000 returns 1110, then translated input 00012300 returns 0111. Consequently, the classification might return differently.

The CNN with max pooling layer is invariant to translation of the foreground object to some extent, which is determined by the specifications of the max pooling layer. In this example, it is not completely invariant.

For example, below is the output after convolutional layer:

![Capture](https://user-images.githubusercontent.com/29167705/63815546-0788c700-c903-11e9-855b-8cf090926fb2.JPG)

If the max pooling layer has 4x4 patch with stride of 4, then the output of max pooling becomes:

![Capture](https://user-images.githubusercontent.com/29167705/63815571-1f604b00-c903-11e9-80fa-16e2434c1c0b.JPG)

We know that convolutional operation is equivariant to translation, so if we translate the foreground object by some pixels, we would get the convolutional output translated by the same number of pixels. if we translate it to any pixels marked in *:

![Capture](https://user-images.githubusercontent.com/29167705/63815611-461e8180-c903-11e9-89f6-987946798a47.JPG)

Then, this is invariant to translation since the output of max pooling would remain the same. However, if we translate the object to the right, say 4 pixels away, the output after convolutional layer becomes:

![Capture](https://user-images.githubusercontent.com/29167705/63815642-651d1380-c903-11e9-9055-9d86a21ad2d9.JPG)

The output after max pooling layer becomes:

![Capture](https://user-images.githubusercontent.com/29167705/63815659-7cf49780-c903-11e9-8779-f06e8b2e4c8f.JPG)

It is no longer invariant to translation since it changes the input to the classifier, which may result in a classification error.
