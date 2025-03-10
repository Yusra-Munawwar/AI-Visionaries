Diabetic Retinopathy Detection using Deep Learning

Introduction
Diabetic Retinopathy (DR) is a severe eye disease caused by diabetes, leading to vision impairment. This project leverages *transfer learning with ResNet152V2* to classify DR images and aid in early detection.

Methodology
1. Data Collection & Preprocessing
- *Dataset*: Retinal images categorized into five classes (No DR, Mild, Moderate, Severe, Proliferative DR).
- *Preprocessing*:
  - Rescaling (normalize pixel values)
  - Resizing to (224, 224)
  - Image Augmentation: rotation, flipping, shear transformation, zooming.

2. Model Architecture
- *Base Model*: ResNet152V2 (pre-trained on ImageNet)
- *Fine-tuning*: Last 10 layers unfrozen for DR-specific feature learning.
- *Additional Layers*:
  - Global Average Pooling
  - Batch Normalization
  - Fully Connected Layers with ReLU activation
  - Dropout (to prevent overfitting)
  - Softmax output for multi-class classification

3. Model Training
- *Optimizer*: Adam
- *Loss Function*: Categorical Cross-Entropy
- *Callbacks*:
  python
  early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
  model.fit(train_generator, validation_data=val_generator, epochs=100, callbacks=[early_stopping, reduce_lr])
  

 4. Model Evaluation
- *Metrics*: Accuracy, AUC, Precision, Recall, F1-score
- *Visualization*: Training/validation accuracy & loss graphs
- *Confusion Matrix*: Analyzes classification errors

Results
- *Training Accuracy*: 80.66%
- *Validation Accuracy*: 80.75%
- *Test Accuracy*: 80.61%
- *AUC Score*: Demonstrates reliability in DR severity classification

Future Improvements
- Fine-tune additional layers for better feature extraction
- Increase dataset size for improved generalization
- Explore attention mechanisms for better interpretability

Conclusion
This deep learning model effectively classifies diabetic retinopathy severity using transfer learning. It offers a promising approach for *early diagnosis and medical intervention*.


