import tensorflow as tf
import argparse, sys
from TS_datasets import getBlood
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import classify_with_knn, interp_data, mse_and_corr, dim_reduction_plot
import math

dim_red = 0
plot_on = 1
interp_on = 0
tied_weights = 0
lin_dec = 1

# parse input data
parser = argparse.ArgumentParser()
parser.add_argument("--code_size", default=20, help="size of the code", type=int)
parser.add_argument("--w_reg", default=0.001, help="weight of the regularization in the loss function", type=float)
parser.add_argument("--a_reg", default=0.1, help="weight of the kernel alignment", type=float)
parser.add_argument("--num_epochs", default=5000, help="number of epochs in training", type=int)
parser.add_argument("--batch_size", default=25, help="number of samples in each batch", type=int)
parser.add_argument("--max_gradient_norm", default=1.0, help="max gradient norm for gradient clipping", type=float)
parser.add_argument("--learning_rate", default=0.001, help="Adam initial learning rate", type=float)
parser.add_argument("--hidden_size", default=30, help="size of the code", type=int)
args = parser.parse_args()
print(args)

# ================= DATASET =================
(train_data, train_labels, train_len, _, K_tr,
        valid_data, _, valid_len, _, K_vs,
        test_data_orig, test_labels, test_len, _, K_ts) = getBlood(kernel='ideal', inp='zero') # data shape is [T, N, V] = [time_steps, num_elements, num_var]

# sort test data (for visualize the learned K)
sort_idx = np.argsort(test_labels,axis=0)[:,0]
test_data_orig = test_data_orig[:,sort_idx,:]
test_labels = test_labels[sort_idx,:]
K_ts = K_ts[sort_idx,:] [:,sort_idx]
       
# interpolation
if np.min(train_len) < np.max(train_len) and interp_on:
    print('-- Data Interpolation --')
    train_data = interp_data(train_data, train_len)
    valid_data = interp_data(valid_data, valid_len)
    test_data = interp_data(test_data_orig, test_len)
else:
    test_data = test_data_orig

# transpose and reshape [T, N, V] --> [N, T, V] --> [N, T*V]
train_data = np.transpose(train_data,axes=[1,0,2])
train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
valid_data = np.transpose(valid_data,axes=[1,0,2])
valid_data = np.reshape(valid_data, (valid_data.shape[0], valid_data.shape[1]*valid_data.shape[2]))
test_data = np.transpose(test_data,axes=[1,0,2])
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))   

print('\n**** Processing Blood data: Tr{}, Vs{}, Ts{} ****\n'.format(train_data.shape, valid_data.shape, test_data.shape))

input_length = train_data.shape[1] # same for all inputs

# ================= GRAPH =================

# init session
tf.reset_default_graph() # needed when working with iPython
sess = tf.Session()

# placeholders
encoder_inputs = tf.placeholder(shape=(None,input_length), dtype=tf.float32, name='encoder_inputs')
prior_K = tf.placeholder(shape=(None, None), dtype=tf.float32, name='prior_K')

# ----- ENCODER -----
We1 = tf.Variable(tf.random_uniform((input_length, args.hidden_size), -1.0 / math.sqrt(input_length), 1.0 / math.sqrt(input_length)))
We2 = tf.Variable(tf.random_uniform((args.hidden_size, args.code_size), -1.0 / math.sqrt(args.hidden_size), 1.0 / math.sqrt(args.hidden_size)))

be1 = tf.Variable(tf.zeros([args.hidden_size]))
be2 = tf.Variable(tf.zeros([args.code_size]))

hidden_1 = tf.nn.tanh(tf.matmul(encoder_inputs, We1) + be1)
code = tf.nn.tanh(tf.matmul(hidden_1, We2) + be2)

# kernel on codes
code_K = tf.tensordot(code, tf.transpose(code), axes=1)

# ----- DECODER -----
if tied_weights:
    Wd1 = tf.transpose(We2)
    Wd2 = tf.transpose(We1)
else:
    Wd1 = tf.Variable(tf.random_uniform((args.code_size, args.hidden_size), -1.0 / math.sqrt(args.code_size), 1.0 / math.sqrt(args.code_size)))
    Wd2 = tf.Variable(tf.random_uniform((args.hidden_size, input_length), -1.0 / math.sqrt(args.hidden_size), 1.0 / math.sqrt(args.hidden_size)))
    
bd1 = tf.Variable(tf.zeros([args.hidden_size]))  
bd2 = tf.Variable(tf.zeros([input_length])) 

if lin_dec:
    hidden_2 = tf.matmul(code, Wd1) + bd1
else:
    hidden_2 = tf.nn.tanh(tf.matmul(code, Wd1) + bd1)

dec_out = tf.matmul(hidden_2, Wd2) + bd2

# ----- LOSS -----
# kernel alignment loss with normalized Frobenius norm
code_K_norm = code_K/tf.norm(code_K, ord='fro', axis=[-2,-1])
prior_K_norm = prior_K/tf.norm(prior_K, ord='fro', axis=[-2,-1])
k_loss = tf.norm(code_K_norm - prior_K_norm, ord='fro', axis=[-2,-1])

# reconstruction loss    
parameters = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(args.learning_rate)
reconstruct_loss = tf.losses.mean_squared_error(labels=dec_out, predictions=encoder_inputs)

# L2 loss
reg_loss = 0
for tf_var in tf.trainable_variables():
    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
        
tot_loss = reconstruct_loss + args.w_reg*reg_loss + args.a_reg*k_loss

# Calculate and clip gradients
gradients = tf.gradients(tot_loss, parameters)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, args.max_gradient_norm)
update_step = optimizer.apply_gradients(zip(clipped_gradients, parameters))

sess.run(tf.global_variables_initializer())

# trainable parameters count
total_parameters = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
        variable_parametes *= dim.value
    total_parameters += variable_parametes
print('Total parameters: {}'.format(total_parameters))

# ============= TENSORBOARD =============             
mean_grads = tf.reduce_mean([tf.reduce_mean(grad) for grad in gradients])
tf.summary.scalar('mean_grads', mean_grads)
tf.summary.scalar('reconstruct_loss', reconstruct_loss)
tvars = tf.trainable_variables()
for tvar in tvars:
    tf.summary.histogram(tvar.name.replace(':','_'), tvar)
merged_summary = tf.summary.merge_all()

# ================= TRAINING =================

# initialize training stuff
batch_size = args.batch_size
time_tr_start = time.time()
max_batches = train_data.shape[0]//batch_size
loss_track = []
kloss_track = []
min_vs_loss = np.infty
model_name = "/tmp/tkae_models/m_"+str(time.strftime("%Y%m%d-%H%M%S"))+".ckpt"
#train_writer = tf.summary.FileWriter('/tmp/tensorboard', graph=sess.graph)
saver = tf.train.Saver()

try:
    for ep in range(args.num_epochs):
        
        # shuffle training data
        idx = np.random.permutation(train_data.shape[0])
        train_data_s = train_data[idx,:] 
        K_tr_s = K_tr[idx,:][:,idx]
        
        for batch in range(max_batches):
            
            fdtr = {encoder_inputs: train_data_s[(batch)*batch_size:(batch+1)*batch_size,:],
                    prior_K: K_tr_s[(batch)*batch_size:(batch+1)*batch_size, (batch)*batch_size:(batch+1)*batch_size]
                    }           
            _,train_loss, train_kloss = sess.run([update_step, reconstruct_loss, k_loss], fdtr)    
            loss_track.append(train_loss)
            kloss_track.append(train_kloss)
            
        # check training progress on the validations set (in blood data valid=train) 
        if ep % 100 == 0:            
            print('Ep: {}'.format(ep))
            
            fdvs = {encoder_inputs: valid_data,
                    prior_K: K_vs}
            outvs, lossvs, klossvs, vs_code_K = sess.run([dec_out, reconstruct_loss, k_loss, code_K], fdvs) #summary, merged_summary
#            plt.matshow(vs_code_K,cmap='binary_r')
#            plt.show()
            #train_writer.add_summary(summary, ep)
            print('VS r_loss=%.3f, k_loss=%.3f -- TR r_loss=%.3f, k_loss=%.3f'%(lossvs, klossvs, np.mean(loss_track[-100:]), np.mean(kloss_track[-100:])))     
            
            # Save model yielding best results on validation
            if lossvs < min_vs_loss:
                min_vs_loss = lossvs
                tf.add_to_collection("encoder_inputs",encoder_inputs)
                tf.add_to_collection("dec_out",dec_out)
                tf.add_to_collection("reconstruct_loss",reconstruct_loss)
                save_path = saver.save(sess, model_name)
                                                    
except KeyboardInterrupt:
    print('training interrupted')

#if plot_on:
#    plt.plot(kloss_track, label='kloss_track')
#    plt.legend(loc='upper right')
#    plt.show(block=False)
    
time_tr_end = time.time()
print('Tot training time: {}'.format((time_tr_end-time_tr_start)//60) )

# ================= TEST =================
print('************ TEST ************ \n>>restoring from:'+model_name+'<<')

tf.reset_default_graph() # be sure that correct weights are loaded
saver.restore(sess, model_name)

tr_code = sess.run(code, {encoder_inputs: train_data})
pred, pred_loss, ts_code, ts_code_K = sess.run([dec_out, reconstruct_loss, code, code_K], {encoder_inputs: test_data})
print('Test loss: %.3f'%(np.mean((pred-test_data)**2)))

# reverse transformations
pred = np.reshape(pred, (test_data_orig.shape[1], test_data_orig.shape[0], test_data_orig.shape[2]))
pred = np.transpose(pred,axes=[1,0,2])
test_data = test_data_orig

if np.min(train_len) < np.max(train_len) and interp_on:
    print('-- Reverse Interpolation --')
    pred = interp_data(pred, test_len, restore=True)

if plot_on:
    
    # plot the reconstruction of a random time series
    plot_idx1 = np.random.randint(low=0,high=test_data.shape[0])
    target = test_data[:,plot_idx1,0]
    ts_out = pred[:,plot_idx1,0]
    plt.plot(target, label='target')
    plt.plot(ts_out, label='pred')
    plt.legend(loc='upper right')
    plt.show(block=True)  
    np.savetxt('AE_pred',ts_out)
    
    # plot the first 2 components of the code
    plt.scatter(ts_code[:,0],ts_code[:,1],c=test_labels, s=80,marker='.',linewidths = 0,cmap='Paired')
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.show()
    
    plt.matshow(K_ts,cmap='binary_r')
    plt.show()
    plt.matshow(ts_code_K,cmap='binary_r')
    plt.show()

# MSE and corr
test_mse, test_corr = mse_and_corr(test_data, pred, test_len)
print('Test MSE: %.3f\nTest Pearson correlation: %.3f'%(test_mse, test_corr))

# kNN classification on the codes
acc, f1, auc = classify_with_knn(tr_code, train_labels[:, 0], ts_code, test_labels[:, 0], k=3)
print('kNN -- acc: %.3f, F1: %.3f, AUC: %.3f'%(acc, f1, auc))

# dim reduction plots
if dim_red:
    dim_reduction_plot(ts_code, test_labels, 1)

#train_writer.close()
sess.close()
