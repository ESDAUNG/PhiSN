import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf

def adap_data(data:tf.data.Dataset.from_tensor_slices,max_len,max_vocab):
    vectorize_layer = preprocessing.TextVectorization(standardize=None,split="whitespace",output_mode="int",pad_to_max_tokens=True,output_sequence_length=max_len,max_tokens=max_vocab)
    vectorize_layer.adapt(data.batch(32))

    return vectorize_layer

def vectorize_data(data:list,vectorize_layer:preprocessing.TextVectorization,batch_size=10):
    tf.keras.backend.clear_session()
    vectorized_id=[]
    first,last=0,int(len(data)/batch_size)
    if len(data)%batch_size==0:
        steps=batch_size
    else:
        steps=batch_size+1
    for i in range(steps):
        if len(data)%batch_size==0:
            vectorized_id=vectorized_id+vectorize_layer(np.array(data[first:last])).numpy().tolist()
            first=last
            last=last+int(len(data)/batch_size)
        else:
            if i<batch_size:
                vectorized_id=vectorized_id+vectorize_layer(np.array(data[first:last])).numpy().tolist()
                first=last
                last=last+int(len(data)/batch_size)
            else:
                last=first+len(data)%batch_size
                vectorized_id=vectorized_id+vectorize_layer(np.array(data[first:last])).numpy().tolist()

    vectorize_id_np=np.array(vectorized_id)
    return vectorize_id_np
