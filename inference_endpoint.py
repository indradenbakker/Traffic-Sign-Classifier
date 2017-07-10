import os

import numpy as np
import tensorflow as tf
import json
import csv
import pickle
import cv2

from PIL import Image
import urllib.request
import io

from flask import Flask, jsonify, request, redirect, url_for
from werkzeug import secure_filename

app = Flask(__name__) # create a Flask app

def normalize_grayscale(image):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    *param image: The image data to be normalized
    *return: Normalized image data
    """
    a = 0.1
    b = 0.9
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

def grayscale(image):
    """
    Applies the Grayscale transform from cv2
    *param image: The image data to be converted to grayscale
    *return: Grayscale image data
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

labels = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# Specify model
x = tf.placeholder(tf.float32, shape=[None, 32*32])
y_ = tf.placeholder(tf.float32, shape=[None, 43])
W = tf.Variable(tf.zeros([32*32,43]))
b = tf.Variable(tf.zeros([43]))

# Layer 1: convolutional
W_conv1 = weight_variable([7, 7, 1, 100])
b_conv1 = bias_variable([100])
x_image = tf.reshape(x, [-1,32,32,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# Layer 2: max pooling
h_pool1 = max_pool_2x2(h_conv1)

# Layer 3: convolutional
W_conv2 = weight_variable([4, 4, 100, 150])
b_conv2 = bias_variable([150])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# Layer 4: max pooling
h_pool2 = max_pool_2x2(h_conv2)

# Layer 5: convolutional
W_conv3 = weight_variable([4, 4, 150, 250])
b_conv3 = bias_variable([250])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

# Layer 6: max pooling
h_pool3 = max_pool_2x2(h_conv3)

# Layer 7: fully connected
W_fc1 = weight_variable([4 * 4 * 250, 300])
b_fc1 = bias_variable([300])

h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*250])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

# Layer 8: dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Layer 9: fully connected
W_fc2 = weight_variable([300, 43])
b_fc2 = bias_variable([43])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
saver = tf.train.Saver()

batch_size = 150

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
predictions_prob = tf.nn.softmax(y_conv)
prediction = tf.argmax(y_conv, 1)

def eval_model_on_dataset(img, labels, ckpt_file):
    saver = tf.train.Saver()
    img_norm = normalize_grayscale(grayscale(img)).flatten()
    with tf.Session() as sess:
        # Restore model checkpoint. 
        saver.restore(sess, ckpt_file)
        for i in range(1):
            y_hat = prediction.eval(feed_dict={x: [img_norm], y_: labels, keep_prob: 1.0})
    return str(class_names[y_hat])


def url_to_image(url, size):
    
    with urllib.request.urlopen(url) as url:
        f = io.BytesIO(url.read())
    img = Image.open(f)
    #data = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    img = img.resize((size, size), Image.ANTIALIAS)
    #data = cv2.resize(img, (size, size))[:,:,0].flatten()
    img = np.asarray(img)
    return img

@app.route('/predict/<path:url>', methods=['POST'])
def predict(url):
    img = url_to_image(url, 32)
    prediction = eval_model_on_dataset(img, [labels], '/home/carnd/CarND-LeNet-Lab/model.ckpt')
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    with open('./signnames.csv', 'r') as f:
        creader = csv.reader(f)
        names = list(creader)
        class_names = [a[1] for a in names[1:]]
    app.run(port=5000)