import string
import tensorflow as tf
import numpy as np
import getConfig
import tensorflow.keras.preprocessing.sequence as sequence
import textClassiferModelMultiReshape as model
import time
import os

def cal_ks(y_true, y_prob, n_bins=20):
    percentile = np.linspace(0, 100, n_bins + 1).tolist()
    bins = [np.percentile(y_prob, i) for i in percentile]
    bins[0] = bins[0] - 0.01
    bins[-1] = bins[-1] + 0.01
    binids = np.digitize(y_prob, bins) - 1
    y_1 = sum(y_true == 1)
    y_0 = sum(y_true == 0)
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    bin_false = bin_total - bin_true
    true_pdf = bin_true / y_1
    false_pdf = bin_false / y_0
    true_cdf = np.cumsum(true_pdf)
    false_cdf = np.cumsum(false_pdf)
    ks_list = np.abs(true_cdf - false_cdf).tolist()
    ks = max(ks_list)
    return ks, ks_list

gConfig = getConfig.get_config(config_file='config.ini.all3')
sentence_size = gConfig['sentence_size']
embedding_size = gConfig['embedding_size']
vocab_size = gConfig['vocabulary_size']
model_dir = gConfig['model_dir']

def read_npz(data_file):
    r = np.load(data_file,allow_pickle=True)
    return r['arr_0'],r['arr_1'],r['arr_2'],r['arr_3'],r['arr_4'],r['arr_5'],r['arr_6'],r['arr_7']

def pad_sequences(inp):
    out_sequences=sequence.pad_sequences(inp,maxlen=gConfig['sentence_size'],padding='post',value=0)
    return out_sequences

def get_alldata_label(file):
    x1, x2, x3, y = [], [], [], []
    with open(file) as f:
        for line in f:
            lines = line.rstrip('\n').split('\t')
            y.append(lines[0].split(',')[2])
            x1.append(lines[0].split(',')[3:44])
            x2.append(lines[0].split(',')[44:])
            x3.append(lines[-2].split(','))
    return np.array(x1), np.array(x2), np.array(x3), np.array(y)

checkpoint_path = gConfig['model_dir']
ckpt_manager = tf.train.CheckpointManager(model.ckpt,checkpoint_path,max_to_keep=15)

def create_model(best_ckpt=None):
    ckpt=tf.io.gfile.listdir(checkpoint_path)
    if ckpt:
        print('reload model')
        if best_ckpt:
            model.ckpt.restore(checkpoint_path+'ckpt-'+str(best_ckpt))
            print('load trained model ckpt-'+str(best_ckpt))
            return model
        else:
            model.ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
            return model
    else:
        return model

def train():
    model = create_model()
    for epoch in range(gConfig['epochs']):
        start = time.time()
        model.train_loss.reset_states()
        model.train_accuracy.reset_states()
        for (batch,(inp1,inp2,inp3,target)) in enumerate(dataset_train.batch(gConfig['batch_size'])):
            start = time.time()
            loss = model.step(inp1,inp2,inp3,target)
            print('train set:Epoch {} Batch {} Loss {:.4f} AUC {:.4f} prestep {:.4f}'.format(epoch+1,batch,loss[0],loss[1],(time.time()-start)))

        for (batch,(inp1,inp2,inp3,target)) in enumerate(dataset_test.batch(gConfig['batch_size'])):
            start = time.time()
            loss = model.evaluate(inp1,inp2,inp3,target)
            print('validate set:Epoch {} Batch {} Loss {:.4f} AUC {:.4f} prestep {:.4f}'.format(epoch+1,batch,loss[0],loss[1],(time.time()-start)))

        ckpt_save_path=ckpt_manager.save()
        print('save epoch{} model at {}'.format(epoch+1,ckpt_save_path))
        files = ['202001_mob2.csv','202002_mob2.csv']
        for file in files:
            X1, X2, X3, Y, IDS = get_data_label_id(gConfig['data_path'] + file)
            X1 = X1.astype(float)
            X2 = X2.astype(float)
            X3 = pad_sequences(X3)
            Y = [int(i) for i in Y]
            dataset_pred = tf.data.Dataset.from_tensor_slices((X1,X2,X3,Y))
            output = []
            for (batch,(inp1,inp2,inp3,target)) in enumerate(dataset_pred.batch(gConfig['batch_size'])):
                data = model.prediction(inp1,inp2,inp3,target)
                if len(output)==0:
                    output = data
                else:
                    output = np.concatenate((output,data),axis=0)
            print(file,' ',cal_ks(output[:,0],output[:,1])[0])

def get_data_label_id(file):
    x1, x2, x3, y ,ids = [], [], [], [], []
    with open(file) as f:
        for line in f:
            lines = line.rstrip('\n').split('\t')
            ids.append(lines[0].split(',')[:2])
            y.append(lines[0].split(',')[2])
            x1.append(lines[0].split(',')[3:44])
            x2.append(lines[0].split(',')[44:])
            x3.append(lines[-2].split(','))
    return np.array(x1), np.array(x2), np.array(x3), np.array(y), ids



def prediction():
    model = create_model(best_ckpt=5)
    files = os.listdir(gConfig['data_path'])
    files = [f for f in files if f[-8:]=='mob2.csv']
    print(files)
    for file in files:
        X1, X2, X3, Y, IDS = get_data_label_id(gConfig['data_path'] + file)
        X1 = X1.astype(float)
        X2 = X2.astype(float)
        X3 = pad_sequences(X3)
        Y = [int(i) for i in Y]
        dataset_pred = tf.data.Dataset.from_tensor_slices((X1,X2,X3,Y))
        output = []
        for (batch,(inp1,inp2,inp3,target)) in enumerate(dataset_pred.batch(gConfig['batch_size'])):
            data = model.prediction(inp1,inp2,inp3,target)
            if len(output)==0:
                output = data
            else:
                output = np.concatenate((output,data),axis=0)
        #output = np.concatenate((IDS,output),axis=1)
        print(file,' ',cal_ks(output[:,0],output[:,1])[0])
        np.savetxt(gConfig['data_path'] + file +'_prediction_all3',output,delimiter=',')


#if __name__ == "__main__":
if gConfig['mode'] == 'train':
    x_train1,x_train2,x_train3, y_train = get_alldata_label('data/train.csv')
    x_test1,x_test2,x_test3, y_test = get_alldata_label('data/test.csv')
    x_train3 = pad_sequences(x_train3)
    x_test3 = pad_sequences(x_test3)
    x_train1 = x_train1.astype(float)
    x_train2 = x_train2.astype(float)
    x_test1 = x_test1.astype(float)
    x_test2 = x_test2.astype(float)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train1,x_train2,x_train3,y_train)).shuffle(gConfig['shuffle_size'])
    dataset_test = tf.data.Dataset.from_tensor_slices((x_test1,x_test2,x_test3,y_test)).shuffle(gConfig['shuffle_size'])
    train()
elif gConfig['mode'] == 'prediction':
    prediction()
