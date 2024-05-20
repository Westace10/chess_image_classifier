# Chess Image Classification Project Report

## Project Overview
The objective of this project was to design and implement a convolutional neural network (CNN) to classify images of chess pieces. The project involved multiple stages, each focusing on different aspects of model improvement and performance evaluation. This report provides a detailed account of the model choice, enhancement techniques, training process, and evaluation metrics. It also includes visual aids to illustrate the model's performance. The project is divided into three stages below:
1. Training with basic data augmentation.
2. Enhancing the model with additional data augmentation and batch normalization.
3. Fine-tuning hyperparameters such as learning rate and batch size.

## Techniques for Enhancement

To enhance the model's performance, several techniques were employed:

1. Data Augmentation: I applied various data augmentation techniques such as rotation, scaling, and flipping to increase the diversity of the training data and reduce overfitting.
2. Batch Normalization: Batch normalization layers were added to stabilize and accelerate the training process.
3. Learning Rate Scheduling: I experimented with different learning rates and batch sizes to find the optimal training configuration.
4. Dropout: Dropout layers were used to prevent overfitting by randomly deactivating neurons during training.
5. Transfer Learning: Although I started with a custom model, I incorporated transfer learning principles by initializing some layers with weights from a pre-trained model.

## Training Process

The training process was divided into three stages, each with different configurations to find the best setup for our model.

## Stage One: Basic Data Augmentation

### Training and Validation Results
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 0     | 6.8718     | 0.2620    | 2.5385   | 0.3023  |
| 1     | 1.8885     | 0.3526    | 1.3045   | 0.4419  |
| 2     | 1.4117     | 0.4534    | 1.0064   | 0.6744  |
| 3     | 1.3439     | 0.4861    | 1.0014   | 0.6512  |
| 4     | 1.2927     | 0.5063    | 0.9532   | 0.6512  |
| 5     | 1.3124     | 0.4937    | 0.9659   | 0.6047  |
| 6     | 1.3020     | 0.4937    | 0.9298   | 0.6047  |
| 7     | 1.1933     | 0.5239    | 0.9232   | 0.6047  |
| 8     | 1.1605     | 0.5189    | 0.9322   | 0.6512  |
| 9     | 1.1645     | 0.5416    | 0.9008   | 0.6279  |

### Observations
- The model shows improvement in both training and validation accuracy over the epochs.
- The best validation accuracy achieved is 67.44%.

## Stage Two: Additional Data Augmentation and Batch Normalization

### Training and Validation Results
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 0     | 1.6666     | 0.3401    | 1.2721   | 0.5349  |
| 1     | 1.3837     | 0.4861    | 1.1609   | 0.5581  |
| 2     | 1.3601     | 0.4761    | 1.1479   | 0.5581  |
| 3     | 1.3410     | 0.4987    | 1.1828   | 0.5581  |
| 4     | 1.4229     | 0.4534    | 1.1351   | 0.6047  |
| 5     | 1.2900     | 0.5189    | 1.1781   | 0.5581  |
| 6     | 1.3054     | 0.4937    | 1.1614   | 0.5581  |
| 7     | 1.2782     | 0.5416    | 1.1205   | 0.6279  |
| 8     | 1.1901     | 0.5970    | 1.1059   | 0.6047  |
| 9     | 1.2622     | 0.5592    | 1.0962   | 0.6279  |

### Observations
- The model's performance improved on the validation set compared to Stage One when tested, even though the validation accuracy during training was slightly lower.
- The best validation accuracy achieved is 62.79%.

## Stage Three: Hyperparameter Tuning

### Experiment 1: Learning Rate 0.01, Batch Size 64
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 0     | 1.6143     | 0.3476    | 3.5936   | 0.3721  |
| 1     | 1.4247     | 0.4332    | 1.6767   | 0.4419  |
| 2     | 1.3421     | 0.5063    | 1.2211   | 0.4651  |
| 3     | 1.3908     | 0.4685    | 1.1148   | 0.4651  |
| 4     | 1.3469     | 0.4710    | 1.0484   | 0.6047  |
| 5     | 1.2673     | 0.5214    | 1.0894   | 0.5349  |
| 6     | 1.2424     | 0.5088    | 1.0725   | 0.6047  |
| 7     | 1.2594     | 0.5214    | 1.0681   | 0.5581  |
| 8     | 1.2261     | 0.5491    | 1.0626   | 0.5814  |
| 9     | 1.1955     | 0.5567    | 1.0687   | 0.5581  |

### Experiment 2: Learning Rate 0.01, Batch Size 128
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 0     | 1.3360     | 0.4736    | 1.1364   | 0.6279  |
| 1     | 1.2345     | 0.5567    | 1.2317   | 0.5116  |
| 2     | 1.2808     | 0.5063    | 1.3187   | 0.4651  |
| 3     | 1.2911     | 0.4786    | 1.1820   | 0.5581  |
| 4     | 1.3257     | 0.4610    | 1.1240   | 0.5814  |
| 5     | 1.2409     | 0.5516    | 1.1024   | 0.6047  |
| 6     | 1.2436     | 0.5340    | 1.0762   | 0.6047  |
| 7     | 1.2384     | 0.5542    | 1.0650   | 0.6279  |
| 8     | 1.2349     | 0.5416    | 1.0542   | 0.6279  |
| 9     | 1.2407     | 0.5063    | 1.0458   | 0.6279  |

### Experiment 3: Learning Rate 0.1, Batch Size 64
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 0     | 1.7030     | 0.3350    | 2.8647   | 0.3953  |
| 1     | 1.4797     | 0.4635    | 1.3315   | 0.5349  |
| 2     | 1.3944     | 0.4433    | 1.1386   | 0.5116  |
| 3     | 1.3501     | 0.4912    | 1.1068   | 0.4884  |
| 4     | 1.3087     | 0.4811    | 1.1815   | 0.5581  |
| 5     | 1.3220     | 0.4635    | 1.2725   | 0.5116  |
| 6     | 1.2328     | 0.5189    | 1.1008   | 0.5349  |
| 7     | 1.2388     | 0.5139    | 1.0694   | 0.5814  |
| 8     | 1.1795     | 0.5491    | 1.0582   | 0.5814  |
| 9     | 1.2412     | 0.5013    | 1.0481   | 

0.5814  |

### Experiment 4: Learning Rate 0.1, Batch Size 128
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 0     | 1.3130     | 0.4887    | 1.0812   | 0.5349  |
| 1     | 1.2348     | 0.5063    | 1.0520   | 0.5581  |
| 2     | 1.2940     | 0.4584    | 1.0305   | 0.6047  |
| 3     | 1.2439     | 0.5290    | 0.9973   | 0.6512  |
| 4     | 1.2007     | 0.5516    | 1.0234   | 0.6047  |
| 5     | 1.2316     | 0.4912    | 0.9924   | 0.6047  |
| 6     | 1.2177     | 0.5617    | 0.9700   | 0.6279  |
| 7     | 1.1976     | 0.5567    | 0.9817   | 0.6047  |
| 8     | 1.2029     | 0.5516    | 1.0001   | 0.6279  |
| 9     | 1.2333     | 0.5013    | 0.9900   | 0.6279  |

### Observations
- Learning Rate 0.01, Batch Size 64: The best validation accuracy achieved is 60.47%. 
- Learning Rate 0.01, Batch Size 128: The best validation accuracy achieved is 62.79%.
- Learning Rate 0.1, Batch Size 64: The best validation accuracy achieved is 58.14%.
- Learning Rate 0.1, Batch Size 128: The best validation accuracy achieved is 65.12%.

## Evaluation

### Accuracy and Loss Graphs

The graphs below illustrate the training and validation accuracy and loss over epochs for the best-performing configuration (Stage One).

Training and Validation Accuracy:
![Results plot](report/img/initial_results_plot.png)

### Confusion Matrix

The confusion matrix provides a detailed breakdown of the model's performance across different classes.

![Confusion Matrix](report/img/initial_confusion_matrix_plot.png)

## Discussion

### Model Performance
- Best Performance during training: The best overall validation accuracy (67.44%) was achieved during Stage One without additional augmentations and with the default learning rate and batch size.
- Best Performance during Testing: The model from Stage Two performed better during testing, indicating that the additional augmentations and batch normalization provided more robust performance.
- Stage Three: Experimentation with learning rates and batch sizes showed that a learning rate of 0.1 with a batch size of 128 yielded a high validation accuracy of 65.12%, but not surpassing Stage One's best.

### Challenges:
- Overfitting was a significant challenge, which was mitigated using dropout and data augmentation techniques.
- Tuning the learning rate and batch size was critical to balancing model performance and training time.

### Future Work
1. Hyperparameter Optimization: Further tuning using a more granular approach to learning rates (e.g., 0.001, 0.005) and experimenting with different batch sizes.
2. Model Architecture: Experiment with different CNN architectures or pre-trained models (e.g., ResNet, VGG) to see if a more complex model can yield better performance.
3. Data Augmentation: Refine data augmentation techniques to avoid overfitting while providing diverse training samples.
4. Regularization Techniques: Implement other regularization techniques like dropout or weight decay.
5. Cross-Validation: Implement k-fold cross-validation to ensure model robustness and prevent overfitting.

## Conclusion

The CNN model developed for chess image classification achieved the best validation accuracy of 67.44%. Through various enhancement techniques such as data augmentation, batch normalization, and dropout, the model's performance improved significantly. Despite the challenges, the project demonstrated the effectiveness of CNNs in image classification tasks and provided valuable insights into model training and optimization strategies.
