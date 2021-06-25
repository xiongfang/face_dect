import cv2
import tensorflow as tf
import unet_pp as unet
import numpy as np
import os
import pathlib
import time
import tensorflow.keras as keras
from tensorflow.keras.callbacks import TensorBoard
import random

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_CHANNEL = 3
MARK_NUM = 3

path_root = "E:/PythonApplication1/wrinkle"
data_root = pathlib.Path(path_root)
all_image_paths = list(str(g) for g in data_root.glob('*.jpg'))
all_label_paths = list(str(g) for g in data_root.glob('*.png'))

train_filenames = all_image_paths[:11]
train_labels = all_label_paths[:11]

test_filenames = all_image_paths[11:]
test_labels = all_label_paths[11:]

def load_and_process_img(img_list,labels):
    for index,img_name in enumerate(img_list):
        if not isinstance(img_name,str):
            img_name = img_name.decode("utf-8")
        label_name = labels[index]
        if not isinstance(label_name,str):
            label_name = label_name.decode("utf-8")
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = np.asarray(img,dtype = float)
        img /= 255.0
        img = resize(img,(IMAGE_WIDTH,IMAGE_HEIGHT))
        label = cv2.imread(label_name,cv2.IMREAD_UNCHANGED)
        b,g,r,label = cv2.split(label) #保留一个通道
        label = np.asarray(label,dtype = float)
        label /= 255.0
        label = np.expand_dims(label,-1)
        label = resize(label,(IMAGE_WIDTH,IMAGE_HEIGHT))
        img_list = random_rotate(random_scale([img,label]))
        yield img_list[0],cv2.add(img_list[0],cv2.merge([img_list[1],img_list[1],img_list[1]]))


def resize(img,shape):
    width,height = shape
    h,w,c = img.shape
    img_r = np.zeros((width,height,c))
    if h>w:
        img = cv2.resize(img,(int(w/h*width),height))
        if len(img.shape) == 2:
            img = np.expand_dims(img,-1)
        img_r[:,:img.shape[1],:] = img
    else:
        img = cv2.resize(img,(width,int(height*h/w)))
        if len(img.shape) == 2:
            img = np.expand_dims(img,-1)
        img_r[:img.shape[0],:,:] = img
    
    return img_r

def random_scale(img_list):
    img_result = []
    rate = random.randint(500,1000)/1000.0
    for img in img_list:
        h,w,c = img.shape
        img = cv2.resize(img,(0,0),fx=rate,fy=rate)
        if len(img.shape) == 2:
            img = np.expand_dims(img,-1)
        img_r = np.zeros((w,h,c))
        img_r[:img.shape[0],:img.shape[1],:] = img
        img_result.append(img_r)
    return img_result
def random_rotate(img_list):
    img_result = []
    angle = random.randint(-1800,1800)/10.0
    for img in img_list:
        h,w,c = img.shape
        center = (w // 2, h // 2)
        # 逆时针-90°(即顺时针90°)旋转图片
        M = cv2.getRotationMatrix2D(center,angle, 1)
        rotated_img = cv2.warpAffine(img, M, (w, h))
        if len(rotated_img.shape) == 2:
            rotated_img = np.expand_dims(rotated_img,-1)
        img_result.append(rotated_img)
    return img_result


train_ds = tf.data.Dataset.from_generator(load_and_process_img,output_types=(tf.float32,tf.float32),output_shapes=((IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL),(IMAGE_WIDTH, IMAGE_HEIGHT, MARK_NUM)),args=[train_filenames,train_labels])
train_ds = train_ds.batch(3)
test_ds = tf.data.Dataset.from_generator(load_and_process_img,output_types=(tf.float32,tf.float32),output_shapes=((IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL),(IMAGE_WIDTH, IMAGE_HEIGHT, MARK_NUM)),args=[test_filenames,test_labels])
test_ds = test_ds.batch(1)

'''
for img,label in train_ds:
    cv2.imshow("img",img[0].numpy())
    cv2.imshow("label",label[0].numpy())
    cv2.waitKey()

for img,label in load_and_process_img(test_filenames,test_labels):
    cv2.imshow("img",img)
    cv2.imshow("label",label)
    cv2.waitKey()
'''

input_shape = (IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNEL)
model = unet.create_segmentation_model(input_shape,MARK_NUM)

model.summary()

name = "tf_test_wk"

# Checkpoint is used to resume training.
checkpoint_dir = os.path.join("E:/checkpoints", name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print("Checkpoint directory created: {}".format(checkpoint_dir))

timeStamp = str(int(time.time()))
log_dir='E:/Logs/'+timeStamp+'/'

# Model built. Restore the latest model if checkpoints are available.
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    print("Checkpoint found: {}, restoring...".format(latest_checkpoint))
    model.load_weights(latest_checkpoint)
    print("Checkpoint restored: {}".format(latest_checkpoint))
else:
    print("Checkpoint not found. Model weights will be initialized randomly.")

# Save a checkpoint. This could be used to resume training.
callback_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, name),
    save_weights_only=True,
    verbose=1,
    save_best_only=False)

callback_tensorboard = TensorBoard(log_dir=log_dir,
                        histogram_freq=1024,
                        write_graph=True,
                        update_freq='batch' #'epoch'
                        )



class LogImages(keras.callbacks.Callback):
    def __init__(self, logdir):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(logdir)
    def on_epoch_end(self, epoch, logs={}):

        img_list = []
        pre_list = []
        true_list = []
        #result_list = []
        for img,label in test_ds:
            img1 = img[0].numpy()
            img_list.append(img1)
            true_list.append(label[0].numpy())
            pre = model(img,False)[0].numpy()
            pre_list.append(pre)
            #_,t = cv2.threshold(pre,0.1,1.0,0)
            #t = np.expand_dims(t,-1)
            #zero = np.zeros(t.shape,t.dtype)
            #mask = cv2.merge([t,t,t])
            #img1 = cv2.add(img1,mask)
            #img1 = cv2.resize(img1,(256,256))
            #result_list.append(img1)

        with self.file_writer.as_default():
            tf.summary.image("img", img_list, step=epoch)
            tf.summary.image("pre", pre_list, step=epoch)
            tf.summary.image("true", true_list, step=epoch)
            #tf.summary.image("r", result_list, step=epoch)

callback_image = LogImages(log_dir)

# List all the callbacks.
callbacks = [callback_checkpoint, callback_tensorboard,callback_image]

model.compile(optimizer='Adam',
              loss=tf.keras.losses.MSE,
              #loss = loss_fn,
              metrics=[tf.keras.metrics.mse])


history = model.fit(train_ds, epochs=500
                    #,steps_per_epoch=steps_per_epoch
                    ,validation_data=test_ds
                    ,verbose=1
                    ,callbacks=callbacks
#                   #,validation_steps=1
                    )

'''
for img,label in train_ds:
    img1 = img[0].numpy()
    pre = model(img,False)[0].numpy()
    _,t = cv2.threshold(pre,0.1,1.0,0)
    t = np.expand_dims(t,-1)
    zero = np.zeros(t.shape,t.dtype)
    mask = cv2.merge([zero,zero,t])
    img1 = cv2.add(img1,mask)
    img1 = cv2.resize(img1,(256,256))
    cv2.imshow("img",img1)
    cv2.waitKey()
'''