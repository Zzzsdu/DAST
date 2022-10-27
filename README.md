# DAST

Requirements
===============
Python 3.5 or above
---------------------

How to run
===============
(1) Data proprecess
Run data_process.py and the data is in Cmapss_data. 

The processed data will be (F001_window_size_trainX.mat, F001_window_size_trainY.mat, F001_window_size_testX.mat, F001_window_size_testY.mat) as F001 example. 

The data processed by sliding time window will be input to Statistical features process.py and will get final data (F001_window_size_trainX_new.mat, F001_window_size_testX_new.mat), the labels are ( F001_window_size_trainY.mat, F001_window_size_testY.mat). The final processed data is too big to upload, you can process it in your computer.

(2) The data preprocessed will be the input to DAST_main.py.

You can load the processed data and pretrained model to get the final results or you can start the model training process and test process. 

Citation
===============
If our code is helpful for your research, please cite our paper:

@article{zhang2022dual,
  title={Dual-Aspect Self-Attention Based on Transformer for Remaining Useful Life Prediction},
  author={Zhang, Zhizheng and Song, Wen and Li, Qiqiang},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={71},
  pages={1--11},
  year={2022},
  publisher={IEEE}
}






