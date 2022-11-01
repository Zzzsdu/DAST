# DAST


How to run
===============
(1) Data proprecess
Run data_process.py and the data is in Cmapss_data. 

The processed data will be (F001_window_size_trainX.mat, F001_window_size_trainY.mat, F001_window_size_testX.mat, F001_window_size_testY.mat) as F001 example. 

The data processed by sliding time window will be input to Statistical features process.py and will get final data (F001_window_size_trainX_new.mat, F001_window_size_testX_new.mat), the labels are ( F001_window_size_trainY.mat, F001_window_size_testY.mat). The final processed data is too big to upload, you can process it in your computer.

(2) The data preprocessed will be the input to DAST model.

You can load the processed data and pretrained model to get the final results.







