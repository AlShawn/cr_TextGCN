"""

remember to change dataset

"""
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
from sklearn import metrics
from utils import *
from models import GCN, MLP
import random
import os
import sys
import func_eval

#---------------------use command to run code block----------------------------------
#if len(sys.argv) != 2:
#	sys.exit("Use: python train.py <dataset>")
#
##datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr','laptop']
#datasets=['laptop_cpu','laptop_hd','laptop_gpu','laptop_screen','laptop_ram']
#dataset = sys.argv[1]
#
#if dataset not in datasets:
#	sys.exit("wrong dataset name")

#if len(sys.argv) != 2:
#	sys.exit("Use: python train.py <dataset>")
#---------------------use command to run block finished----------------------------------

#---------------------use spyder to run code block----------------------------------

#datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr','laptop']
datasets=['laptop_cpu','laptop_hd','laptop_gpu','laptop_screen','laptop_ram']
dataset = datasets[4]

#---------------------use spyder to run block finished-----------------------------------

# Set random seed
seed = random.randint(1, 200)
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

flags = tf.app.flags
FLAGS = flags.FLAGS
# 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('dataset', dataset, 'Dataset string.')
# 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
# originally epochs=200
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0,
                   'Weight for L2 loss on embedding matrix.')  # 5e-4
flags.DEFINE_integer('early_stopping', 15,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    FLAGS.dataset) 
print(adj)
# print(adj[0], adj[1])
features = sp.identity(features.shape[0])  # featureless

print(adj.shape)
print(features.shape)

print("train_mask shape:{} total value:{}".format(train_mask.shape,sum(train_mask)))
print("test_mask shape:{} total value:{}".format(test_mask.shape,sum(test_mask)))
print("val_mask shape:{} total value:{}".format(val_mask.shape,sum(val_mask)))

#sys.exit(0)

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32)
}

# Create model
print(features[2][1])
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=session_conf)


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(
        features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy,
                     model.layers[0].embedding], feed_dict=feed_dict)

    # Validation
    cost, acc, pred, labels, duration = evaluate(
        features, support, y_val, val_mask, placeholders)
    
#    editing the output from top-1 to top-5
#    print("predicted labels in val set:")
#    print(pred)
#    print("shape:",pred.shape)
    
    cost_val.append(cost)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(
              outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, pred, labels, test_duration = evaluate(
    features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

test_pred = []
test_labels = []
print(len(test_mask))
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(labels[i])

#----------- evaluation using test_pred and test_labels---------

print('test_pred length:{}'.format(np.array(test_pred).shape))
print('test_labels length:{}'.format(len(test_labels),test_labels))

print(test_pred)
print(test_labels)

test_pred=np.array(test_pred)
#test_labels=np.array(test_labels)
y_real=[]
for i,each in enumerate(test_labels):
    y_real.append(np.array([each]))
y_real=np.array(y_real)    

print('current exper_param is : ',dataset)

print("ncdg:")
i=0
ndcgs=[]
top_k=5
while i < top_k:
    
    y_pred = test_pred[:, 0:i+1]
    i = i+1

    ndcg_i = func_eval._NDCG_score(y_pred,y_real)
    ndcgs.append(ndcg_i)

    print(ndcg_i)

#sys.exit(0)  

print("precision:")
i = 0
precisions = []

while i < top_k:
    
    y_pred = test_pred[:, 0:i+1]
    i = i+1

    precision = func_eval._precision_score(y_pred,y_real)
    precisions.append(precision)

    print(precision)

print("recall:")
i = 0
recalls = []
while i < top_k:
    
    y_pred = test_pred[:,  0:i+1]

    i = i+1   
    recall = func_eval.new_recall(y_pred, y_real)
    recalls.append(recall)

    print(recall)




#----------------below is commentted------------------

#print("Test Precision, Recall and F1-Score...")
#print(metrics.classification_report(test_labels, test_pred, digits=4))
#print("Macro average Test Precision, Recall and F1-Score...")
#print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
#print("Micro average Test Precision, Recall and F1-Score...")
#print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))
#
## doc and word embeddings
#print('embeddings:')
#word_embeddings = outs[3][train_size: adj.shape[0] - test_size]
#train_doc_embeddings = outs[3][:train_size]  # include val docs
#test_doc_embeddings = outs[3][adj.shape[0] - test_size:]
#
#print(len(word_embeddings), len(train_doc_embeddings),
#      len(test_doc_embeddings))
#print(word_embeddings)
#
#f = open('data/corpus/' + dataset + '_vocab.txt', 'r')
#words = f.readlines()
#f.close()
#
#vocab_size = len(words)
#word_vectors = []
#for i in range(vocab_size):
#    word = words[i].strip()
#    word_vector = word_embeddings[i]
#    word_vector_str = ' '.join([str(x) for x in word_vector])
#    word_vectors.append(word + ' ' + word_vector_str)
#
#word_embeddings_str = '\n'.join(word_vectors)
#f = open('data/' + dataset + '_word_vectors.txt', 'w')
#f.write(word_embeddings_str)
#f.close()
#
#doc_vectors = []
#doc_id = 0
#for i in range(train_size):
#    doc_vector = train_doc_embeddings[i]
#    doc_vector_str = ' '.join([str(x) for x in doc_vector])
#    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
#    doc_id += 1
#
#for i in range(test_size):
#    doc_vector = test_doc_embeddings[i]
#    doc_vector_str = ' '.join([str(x) for x in doc_vector])
#    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
#    doc_id += 1
#
#doc_embeddings_str = '\n'.join(doc_vectors)
#f = open('data/' + dataset + '_doc_vectors.txt', 'w')
#f.write(doc_embeddings_str)
#f.close()
