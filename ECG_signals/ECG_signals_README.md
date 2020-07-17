# "ECG_signal" folder contains various examples of ECG signal processing and predictions. 

* It might contain .py files or .ipynb files. 

## "qt-database-1.0.0.zip" 
* This zip file contains some ECG data sets downloaded from Physionet data base, 			 https://www.physionet.org/content/qtdb/1.0.0/
  It contains information about EMG annotation, segmentation, classification, etc.

## ["ECG_data_annatation.ipynb"](https://github.com/CodyDeepPlay/StudyExamples/blob/master/ECG_signals/ECG_data_annotation.ipynb)  
* This file shows how to prepare the data from Physionet data base, and organize it with ECG raw data and its corresponding annotation.
 
## ["ECG_segmentation.py"](https://github.com/CodyDeepPlay/StudyExamples/blob/master/ECG_signals/ECG_segmentation.py)
* This file shows an example of how to use an lstm based deep learning framwork to conduct ECG signal annotation. 
folder 'lstm_seg_raw20200710-1533' contrained a pre-trained model, this is a lstm based simple deep learning model.
One can use the pre-trained model to conduct ECG segmentation.

Below is the mode information

Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
-----------------------------------------------------------------
lstm_4 (LSTM)                (None, 140, 250)          252000    
_________________________________________________________________
dropout_4 (Dropout)          (None, 140, 250)          0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 140, 250)          1000      
_________________________________________________________________
dense_4 (Dense)              (None, 140, 6)            1506      
-----------------------------------------------------------------
Total params: 254,506
Trainable params: 254,006
Non-trainable params: 500
_________________________________________________________________
 
 
 Here is an example of true signal and its predicted segmentation. 
 
 ![predicted segmentation](https://github.com/CodyDeepPlay/StudyExamples/blob/master/ECG_signals/data_folder.JPG)