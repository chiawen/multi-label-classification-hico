# multi-label-classification-hico

## Requirements
- Python
- NumPy
- Tensorflow 1.0

## Model
Inception V3 + logistic sigmoid layer

## Data
HICO dataset: [HICO version 20150920](http://napoli18.eecs.umich.edu/public_html/data/hico_20150920.tar.gz)

Please extract the above file and store extracted files inside the `./hico_data` directory. <br/>

For the CNN, I use Inception v3, pre-trained on ImageNet.
- [inception\_v3\_2016\_08\_28.tar.gz](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)

Extract the compressed file and put `inception_v3.ckpt` into the `./checkpoints` directory.

## Usage
First, extract the filenames and labels of the training set and the testing set.
	
	$ python process_hico_labels.py

Second, convert the image files and annotations to TFRecords.

	$ python hico_to_tfrecords.py

To fine-tune the last layer of Inception v3 for 10 epochs:

 	$ python finetune.py

To evaluate mAP scores on the testing set:

	$ python eval.py

## Evaluation
|Model|mAP|
|:---:|:---:|
|Inception v3 + finetune|0.263|

## Related works
- Yu-Wei Chao, Zhan Wang, Yugeng He, Jiaxuan Wang, and Jia Deng. HICO: A Benchmark for Recognizing Human-Object Interactions in Images. In ICCV, 2015.
- Arun Mallya and Svetlana Lazebnik. Learning Models for Actions and Person-Object Interactions with Transfer to Question Answering. In ECCV, 2016.
