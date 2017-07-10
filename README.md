# Traffic Sign Classifier

For this project we will be using the [German Traffic Sign dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Specifically, we will be using a pickled version of the dataset provided by [Udacity](www.udacity.com) ([download](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)). The project can be divided in three parts: data exploration, model building, and REST API for inference. Below we describe these three parts in more detail. 

Important dependencies:
- Python 3.5.2
- TensorFlow 1.1.0
- OpenCV 3.2.0.7

For this project, we've used a NVIDIA GTX 1080 Ti GPU to train and test our model. The REST API is deployed and tested on a Amazon AWS P2.xlarge instance.


## Project Description
The goal of this project is to create a traffic sign classifier that is able to predict the correct class of a traffic sign captured in an image. We will review our model based on accuracy and inference time. First we need to analayse the available data, this will give the first insights in the possible solutions and challenges for this project. Next, we split up the training data in a training and validation set for model selection and finally we introduce a basic REST API endpoint for other applications.

### Data exploration (see [Jupyter Notebook](https://github.com/indradenbakker/Traffic-Sign-Classifier/blob/master/notebook/Traffic%20Sign%20Classifier%20-%20Data%20Analysis%20%26%20Model%20Training.ipynb))

The dataset contains a total of 34,799 training images that we can use. All images have a size of 32x32x3 and are cropped. Below an overview with examples of all 43 classes.
![alt tag](https://github.com/indradenbakker/traffic-sign-classifier/blob/master/images/example_training_images.png?raw=true)

We can immediately notice some of the challenges that arise with the given dataset:
* The training dat is perfectly cropped around the traffic signs. This means that our model will most likely have difficulties with new images that aren't cropped. This means that for street scenery, we will first need to apply a localisation model before we can classify the traffic signs or we have to combine both in one model (for example YOLO). 
* The low resolution of the images, will probably result in performance loss. 
* The classes are not evenly distributed.

### Model building (see [Jupyter Notebook](https://github.com/indradenbakker/Traffic-Sign-Classifier/blob/master/notebook/Traffic%20Sign%20Classifier%20-%20Data%20Analysis%20%26%20Model%20Training.ipynb))
We've chosen to apply image augmentations to our training data before feeding them to the network. This will make our model more robust to different inputs. We've also chosen to only use the grayscale of images. This to speed up computations during training, but especially during inference. The reason is that we want to try to combine a high accuracy with real-time performance. 

![alt tag](https://github.com/indradenbakker/Traffic-Sign-Classifier/blob/master/images/training%20results.png?raw=true)

The model consists of 3 convolutional blocks and two fully connected layers. To reduce overfitting, we've applied regularisation techniques like dropout and max pooling. This is a  basic model, and can be tuned further for optimum performance. The goal of this model is to provide a solution that gives a relatively high accuracy (> 97%) on the validation set and should be seen as a stepping stone for further development. 

### REST API (see [script](https://github.com/indradenbakker/Traffic-Sign-Classifier/blob/master/inference_endpoint.py))
To make our model available for other applications, we've created a REST API endpoint in Python with Flask. Currently, the REST API is deployed on a Amazon AWS instance and other applications can send POST request including a URL of an image that included a traffic sign. The endpoint will return the predicted class of the image. For example:

`curl -X POST http://127.0.0.1:5000/predict/https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Spain
_traffic_signal_p2.svg/218px-Spain_traffic_signal_p2.svg.png`

`
{
  "prediction": "Children crossing"
}
`

### Discussion

There is clearly room for improvement. Some ideas to improve the project further:
* Include all channels of the images.
* Add more augmentations and rondomly apply augmentations on the fly instead of loading them in memory. 
* Increase the complexity of the network and add skip layers to speed up inference time.
* Provide a more scalable REST API.
