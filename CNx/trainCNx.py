# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:57:48 2017

@author: hfailmezger
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:11:54 2017

@author: hfailmezger
"""
import tensorflow as tf

#import seaborn as sns
from sklearn.svm import SVC
import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#
import pickle
import time
import pandas as pd
from scipy import stats
from sklearn import svm
from sklearn.model_selection import cross_val_score
from scipy.spatial import cKDTree
import math
from sklearn import mixture
from sklearn.model_selection import LeaveOneOut
from random import randint
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import seaborn as sns
# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


def masking_noise(data, sess, v):
    """Apply masking noise to data in X.
    In other words a fraction v of elements of X
    (chosen at random) is forced to zero.
    :param data: array_like, Input data
    :param sess: TensorFlow session
    :param v: fraction of elements to distort, float
    :return: transformed data
    """
    data_noise = data.copy()
    rand = tf.random_uniform(data.shape)
    data_noise[sess.run(tf.nn.relu(tf.sign(v - rand))).astype(np.bool)] = 0

    return data_noise

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']),name="encoder_op1")
    layer_1_encodedNorm=tf.layers.batch_normalization(layer_1)
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1_encodedNorm, weights['encoder_h2']),
                                   biases['encoder_b2']),name="encoder_op2")
    layer_2_encodedNorm=tf.layers.batch_normalization(layer_2)
    # Encoder Hidden layer with sigmoid activation #3
    #layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
    #                              biases['encoder_b3']))
    # Encoder Hidden layer with sigmoid activation #3
    layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_2_encodedNorm, weights['encoder_h4']),
                                   biases['encoder_b4']),name="encoder_op4")
    layer_4_encodedNorm=tf.layers.batch_normalization(layer_4)
    return layer_4_encodedNorm


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1_decoded = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']),name="decoder_op1")
    layer_1_decodedNorm=tf.layers.batch_normalization(layer_1_decoded)
    # Decoder Hidden layer with sigmoid activation #2
    layer_2_decoded = tf.nn.relu(tf.add(tf.matmul(layer_1_decodedNorm, weights['decoder_h3']),
                                   biases['decoder_b2']),name="decoder_op3")
    layer_2_decodedNorm=tf.layers.batch_normalization(layer_2_decoded)
    # Decoder Hidden layer with sigmoid activation #1
    #layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
    #                               biases['decoder_b3']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_4_decoded = tf.nn.sigmoid(tf.add(tf.matmul(layer_2_decodedNorm, weights['decoder_h4']),
                                   biases['decoder_b4']),name="decoder_op4")
    layer_4_decoded_decodedNorm=tf.layers.batch_normalization(layer_4_decoded)
    return layer_4_decoded_decodedNorm


#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#prefix="R:\Processed\DL-Network\\"
#prefix="/data/rds/DMP/DUDMP/COPAINGE/hfailmezger/Processed/DL-Network/"           
###############################################################################

prefix="R:\Processed\DL-Network\\moreLayers\\"
#prefix="/data/rds/DMP/DUDMP/COPAINGE/hfailmezger/Processed/DL-Network/morelayers/"           
prefixData="R:\\Processed\\SKCM-Summary\\"
#prefixData="/data/rds/DMP/DUDMP/COPAINGE/hfailmezger/Processed/SKCM-Summary/"           
prefixSurvivalData="R:\\ReferenceData\\"
#prefixSurvivalData="/data/rds/DMP/DUDMP/COPAINGE/hfailmezger/ReferenceData/"
#
clinicalData=pd.read_table(prefixSurvivalData+"data_bcr_clinical_data_patient.txt")
colnamesPatient=clinicalData.axes[1].tolist()
clinicalDataPR=clinicalData[['Patient Identifier','Sex','Primary melanoma tumor ulceration']]
#
#geneExpressionTable=pd.read_table(prefix+"GeneExpression-RatioLym.tab")
#geneExpressionTable=pd.read_table("R:\\Processed\\SKCM-Summary\\GeneExpression-CNA-Median.tab")
geneExpressionTable=pd.read_table(prefixData+"GeneExpression-CNA-Median-hg19-binarized-newRun.tab")
patientSurvTimes=pd.read_csv(prefixSurvivalData+"survivalTimes.tab", sep='\t')
patientSurvTimes=pd.concat([patientSurvTimes["bcr_patient_barcode"], patientSurvTimes["times"]], axis=1)
#
rownamesMatrix=geneExpressionTable.axes[0].tolist()
geneExpressionTable['PatientName']=rownamesMatrix
geneExpressionTableM=pd.merge(geneExpressionTable,clinicalDataPR,left_on='PatientName',right_on='Patient Identifier',how='inner')
geneExpressionTableMSurvival=geneExpressionTableM.merge(patientSurvTimes, left_on='PatientName', right_on='bcr_patient_barcode', how='left')
###############################################################################
copyNumberTable=pd.read_table(prefixData+"CNA-DL-Median-2-masked-hg19-newRun.tab")
genesCopyNumberTable=copyNumberTable.axes[1].tolist()
############################################################
#
genderClassT=geneExpressionTableMSurvival['Sex'].as_matrix()
genderClass=genderClassT=="Male"
genderClass=genderClass.astype(int)
#
ulcerationClassT=geneExpressionTableMSurvival['Primary melanoma tumor ulceration'].as_matrix()
ulcerationClass=ulcerationClassT=="YES"
ulcerationClass=ulcerationClass.astype(int)
#
pathClass=geneExpressionTableMSurvival['PathClass'].as_matrix()
clusteringClass=geneExpressionTableMSurvival['ClusteringClass'].as_matrix()
lymClass=geneExpressionTableMSurvival['lymClass'].as_matrix()
timeClass=geneExpressionTableMSurvival['times'].as_matrix()
#
patientsGETable=geneExpressionTableMSurvival['PatientName']
geneExpressionTableMSurvival=geneExpressionTableMSurvival.drop('PatientName', axis=1)
geneExpressionTableMSurvival=geneExpressionTableMSurvival.drop('Patient Identifier', axis=1)
geneExpressionTableMSurvival=geneExpressionTableMSurvival.drop('Sex', axis=1)
geneExpressionTableMSurvival=geneExpressionTableMSurvival.drop('Primary melanoma tumor ulceration', axis=1)
geneExpressionTableMSurvival=geneExpressionTableMSurvival.drop('times', axis=1)
geneExpressionTableMSurvival=geneExpressionTableMSurvival.drop('bcr_patient_barcode', axis=1)
geneExpressionTableMSurvival=geneExpressionTableMSurvival.drop('PathClass', axis=1)
geneExpressionTableMSurvival=geneExpressionTableMSurvival.drop('ClusteringClass', axis=1)
geneExpressionTableMSurvival=geneExpressionTableMSurvival.drop('lymClass', axis=1)
#
geneExpressionTableMSurvivalG = geneExpressionTableMSurvival[genesCopyNumberTable]
colnamesMatrixT=geneExpressionTableMSurvivalG.axes[1].tolist()
#
geneExpressionMatrix=geneExpressionTableMSurvivalG.as_matrix()
colnamesMatrixGeneExpression=geneExpressionTableMSurvivalG.axes[1].tolist()
#geneExpressionMatrix=geneExpressionMatrix[:,1:1000]
row_sums = geneExpressionMatrix.sum(axis=1)
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler.fit(geneExpressionMatrix)
#geneExpressionMatrixScaled=scaler.transform(geneExpressionMatrix)
geneExpressionMatrixScaled=preprocessing.scale(geneExpressionMatrix)
#geneExpressionMatrixScaled = preprocessing.scale(geneExpressionMatrix,axis=1)
###############################################################################
copyNumberTableN=copyNumberTable.loc[patientsGETable,:]
copyNumberMatrix=copyNumberTableN.as_matrix()
colnamesCopyNumberMatrix=copyNumberTable.axes[1].tolist()

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler.fit(copyNumberMatrix)
copyNumberMatrixScaled=scaler.transform(copyNumberMatrix)
copyNumberMatrixScaled=preprocessing.scale(copyNumberMatrix)
#copyNumberMatrix#preprocessing.scale(copyNumberMatrix)
###############################################################################
# Training Parameters
learning_rate = 0.001
num_steps = 10000

display_step = 100
save_step = 500

examples_to_show = 10

# Network Parameters
num_hidden_1 = 10000 # 1st layer num features
num_hidden_2 = 7000 # 2nd layer num features (the latent dim)
#num_hidden_3 = 5000 # 3nd layer num features (the latent dim)
num_hidden_4 = 1000# 4nd layer num features (the latent dim)

num_input =  geneExpressionMatrixScaled.shape[1] # MNIST data input (img shape: 28*28)
#num_input = 784 
# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input],name="input_V")
Y=  tf.placeholder("float", [None, num_input],name="CNA_V")

weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([num_input, num_hidden_1]),name="weight_encoder_h1"),
    'encoder_h2': tf.Variable(tf.truncated_normal([num_hidden_1, num_hidden_2]),name="weight_encoder_h2"),
    #'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3])),
    'encoder_h4': tf.Variable(tf.truncated_normal([num_hidden_2, num_hidden_4]),name="weight_encoder_h4"),
    'decoder_h1': tf.Variable(tf.truncated_normal([num_hidden_4, num_hidden_2]),name="weight_decoder_h1"),
   # 'decoder_h2': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2])),
    'decoder_h3': tf.Variable(tf.truncated_normal([num_hidden_2, num_hidden_1]),name="weight_decoder_h3"),
    'decoder_h4': tf.Variable(tf.truncated_normal([num_hidden_1, num_input]),name="weight_decoder_h4"),
}
biases = {
    'encoder_b1': tf.Variable(tf.truncated_normal([num_hidden_1]),name="bias_encoder_b1"),
    'encoder_b2': tf.Variable(tf.truncated_normal([num_hidden_2]),name="bias_encoder_b2"),
   # 'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3])),
    'encoder_b4': tf.Variable(tf.truncated_normal([num_hidden_4]),name="bias_encoder_b4"),
    'decoder_b1': tf.Variable(tf.truncated_normal([num_hidden_2]),name="bias_decoder_b1"),
    'decoder_b2': tf.Variable(tf.truncated_normal([num_hidden_1]),name="bias_decoder_b2"),
   # 'decoder_b3': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b4': tf.Variable(tf.truncated_normal([num_input]),name="bias_decoder_b4"),
}
#with tf.device('/gpu:0'):
#    # Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = Y

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#
#loss = tf.reduce_sum(y_true * tf.log(y_pred) + (1-y_true*(tf.log(1-y_pred))))
#
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

print("START LEARNING")

# Start Training
# Start a new TF session
sess = tf.Session()

# Run the initializer
sess.run(init)

#reducedDatasetOrig = sess.run(encoder_op, feed_dict={X: geneExpressionMatrixScaled})
saver = tf.train.Saver()
# Training
allLosses=[]

#geneExpressionMatrixScaled=geneExpressionMatrix
for i in range(1, num_steps+1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    indicesBatch=np.random.choice(geneExpressionMatrixScaled.shape[0], geneExpressionMatrixScaled.shape[0],replace=False)
    batch_x=geneExpressionMatrixScaled[indicesBatch,]
    #batch_xNoised=masking_noise(batch_x, sess, 0.1)
    batch_y=copyNumberMatrixScaled[indicesBatch,]
    batch_yNoised=masking_noise(batch_y, sess, 0.1)
    #batch_x, _ = mnist.train.next_batch(batch_size)
    # Run optimization op (backprop) and cost op (to get loss value)
    _, l = sess.run([optimizer, loss], feed_dict={X: batch_yNoised, Y: geneExpressionMatrixScaled[indicesBatch,]})
    allLosses.append(l)
    # Display logs per step
    #print(str(i))
    if i % display_step == 0 or i == 1:
        print('Step %i: Minibatch Loss: %f' % (i, l))
    if i % save_step == 0:
        np.savetxt(prefix+"DL-NetworkModel-CNA-GeneExpression-Median-GELoss-hg19.ckpt-AllLosses-N2-MinMax-DF-2L-7000-NewRun-moreSteps-NewRun-moreLayers-"+str(i)+"-2ndRun.txt", allLosses, fmt='%s', delimiter=',')  
        save_path = saver.save(sess, prefix+"DL-NetworkModel-CNA-GeneExpression-Median-GELoss-hg19-N2-MinMax-DF-2L-7000-NewRun-moreSteps-NewRun-moreLayers-"+str(i)+"-2ndRun.ckpt")
        encoded=  sess.run([encoder_op], feed_dict={X: copyNumberMatrixScaled}) 
        en=encoded[0]
        sns_plot = sns.heatmap(en)
        fig = sns_plot.get_figure()
        fig.savefig(prefix+"heatmap-Encoded"+str(i)+"-2ndRun.png")
        plt.close(fig)
        decoded = sess.run([decoder_op], feed_dict={X: copyNumberMatrixScaled})
        den=decoded[0]
        sns_plot = sns.heatmap(den)
        fig = sns_plot.get_figure()
        fig.savefig(prefix+"heatmap-Decoded"+str(i)+"-2ndRun.png")
        plt.close(fig)

encoded=  sess.run([encoder_op], feed_dict={X: copyNumberMatrixScaled})    
decoded = sess.run([decoder_op], feed_dict={X: copyNumberMatrixScaled})

en=encoded[0]
den=decoded[0]

#
#module load anaconda/3/4.3.0
#source activate tfDavrosGPU1p4


ax = sns.heatmap(en)
ax = sns.heatmap(den)



ax = sns.heatmap(den)

ax = sns.heatmap(np.concatenate((geneExpressionMatrixScaled/np.max(geneExpressionMatrixScaled), np.log(den + 0.00000000000000000001)/np.max(np.log(den+ 0.00000000000000000001))), axis=0))


np.savetxt(prefix+"DL-NetworkModel-CNA-GeneExpression-Median-GELoss-hg19.ckpt-AllLosses-N2-MinMax-DF-2L-7000-NewRun-moreSteps-NewRun-2ndRun.txt", allLosses, fmt='%s', delimiter=',')  
save_path = saver.save(sess, prefix+"DL-NetworkModel-CNA-GeneExpression-Median-GELoss-hg19-N2-MinMax-DF-2L-7000-NewRun-moreSteps-NewRun-2ndRun.ckpt")
##############################     
sess.close()
