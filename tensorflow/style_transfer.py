#--*-- coding:utf-8 --*-- 
import os 

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from time import time 

import tensorflow as tf 
import numpy as np 
import scipy.io 
import scipy.misc

from PIL import Image, ImageOps

class IMAGE_PREPARE(object):

    def gen_noise_image(self, width, height):

        noise_image = np.random.uniform(-20, 20, (1, height, width, 3)).astype(np.float32)

        return noise_image 

    def save_image(self, path, image):

        image = image[0]

        image = np.clip(image, 0, 255).astype('uint8')

        scipy.misc.imsave(path, image)

    def get_resized_image(self, image, width, height, reimage_path = ''):

        image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)

        if reimage_path:

            image_dirs = reimage_path.split('/')

            image_dirs[-1] = 'resized_' + image_dirs[-1]

            out_path = '/'.join(image_dirs)

            if not os.path.exists(out_path):

                image.save(out_path)

        image = np.asarray(image, np.float32)

        return np.expand_dims(image, 0)

class VGG(object):

    def __init__(self, vgg_path, input_image):
        '''
        vgg_path : str
            the path of vgg19's file. 

        input_image : array
            shape is (1, heigth, width, 3)
        '''
        
        vgg = scipy.io.loadmat(vgg_path) 

        self.vgg_layers = vgg['layers'][0]

        self.mean = vgg['normalization'][0][0][0][0][0]

        self.input_image = input_image 

        self.layers = (
            (0, 'conv1_1'), (2, 'conv1_2'), (4, 'pool1'),
            (5, 'conv2_1'), (7, 'conv2_2'), (9, 'pool2'), 
            (10, 'conv3_1'), (12, 'conv3_2'), (14, 'conv3_3'), (16, 'conv3_4'), (18, 'pool3'),
            (19, 'conv4_1'), (21, 'conv4_2'), (23, 'conv4_3'), (25, 'conv4_4'), (27, 'pool4'),
            (28, 'conv5_1'), (30, 'conv5_2'), (32, 'conv5_3'), (34, 'conv5_4'), (36, 'pool5')
        ) 


        self.layers_image_dict = self.get_layers_image(self.input_image)

    def _pool(self, x, layer_name, method='avg'):

        with tf.variable_scope(layer_name) as scope:

            if method == 'avg':

                out = tf.nn.avg_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

            else:

                out = tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        setattr(self, layer_name, out)

        return out

    def _conv2d(self, x, weights, bias, layer_name):

        with tf.variable_scope(layer_name) as scope:

            weights = tf.constant(weights, name = 'weights')

            bias = tf.constant(bias.reshape([-1]), name = 'bias')

            conv2d = tf.nn.conv2d(x, weights, strides = [1, 1, 1, 1], padding = 'SAME')

            out = tf.nn.relu(tf.nn.bias_add(conv2d, bias))

        setattr(self, layer_name, out)

        return out 

    def get_layers_image(self, x):

        layers_image_dict = {'src_image':x}

        for layer_idx, layer_name in self.layers:

            if layer_name[:4] == 'conv':

                weights, bias = self.vgg_layers[layer_idx][0][0][0][0]

                x = self._conv2d(x, weights, bias, layer_name)

            else:

                x = self._pool(x, layer_name)
            
            layers_image_dict[layer_name] = x 

        return layers_image_dict 

class STYLE_TRANSFER(IMAGE_PREPARE):

    def __init__(self, vgg_path, style_image_path, content_image_path, noise_rate = 0.5):

        content_image = Image.open(content_image_path)

        style_image = Image.open(style_image_path)

        width, height = content_image.size

        self.style_image = IMAGE_PREPARE.get_resized_image(self, style_image, width, height)

        self.content_image = IMAGE_PREPARE.get_resized_image(self, content_image, width, height)

        init_image = IMAGE_PREPARE.gen_noise_image(self, width, height)

        self.init_image = noise_rate * init_image + (1 - noise_rate) * self.content_image

        with tf.variable_scope('input_image') as scope:

            self.input_image = tf.get_variable(name = 'input_image', shape = self.content_image.shape, dtype = tf.float32, initializer = tf.zeros_initializer())

        self.vgg_path = vgg_path

        self.content_layers = ('conv4_2', )

        self.style_layers = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1')

        self.style_loss_weights = 1.0

        self.content_loss_weights = 0.1 

        self.style_layers_weights = (0.5, 1.0, 1.5, 3.0, 4.0)

        self.lr = 2.0 

        self.gstep = tf.Variable(0, name='step', dtype=tf.int32, trainable=False)

    def load_vgg(self):

        self.vgg = VGG(self.vgg_path, self.input_image)

        self.content_image -= self.vgg.mean 

        self.style_image -= self.vgg.mean

    def _gram_matrix(self, F, M, N):

        F = tf.reshape(F, (M, N))

        return tf.matmul(tf.transpose(F), F)

    def _content_loss(self, F, P):

        self.content_loss = tf.reduce_sum((F - P) ** 2) / (4.0 * P.size)

    def _single_style_loss(self, a, g):

        M = a.shape[1] * a.shape[2]

        N = a.shape[3]

        gram_a = self._gram_matrix(a, M, N)

        gram_g = self._gram_matrix(g, M, N)

        return tf.reduce_sum((gram_g - gram_a) ** 2/ ((2 * M * N) ** 2))


    def _style_loss(self, A):

        l_A = len(A)

        self.style_loss = sum([self._single_style_loss(A[i], getattr(self.vgg, self.style_layers[i])) * self.style_layers_weights[i] for i in range(l_A)])


    def total_loss(self):

        with tf.variable_scope('total_loss') as scope:

            with tf.Session() as sess:

                gen_image_content = getattr(self.vgg, self.content_layers[0])

                sess.run(self.input_image.assign(self.content_image))

                content_image_content = sess.run(gen_image_content)

            self._content_loss(gen_image_content, content_image_content)

            with tf.Session() as sess:

                sess.run(self.input_image.assign(self.style_image))

                style_layers = sess.run([getattr(self.vgg, layer_name) for layer_name in self.style_layers])

            self._style_loss(style_layers)

            self.loss = self.content_loss_weights * self.content_loss + self.style_loss_weights * self.style_loss 

    def optimize(self):

        self.opt = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.loss, global_step = self.gstep)

    def build(self):

        self.load_vgg()

        self.total_loss()

        self.optimize()

    def train(self, max_iters):

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            sess.run(self.input_image.assign(self.init_image))

            start_time = time()

            skip_step = 20 

            for idx in range(max_iters):

                sess.run(self.opt)

                if (idx + 1) % skip_step == 0 and idx != 0:

                    gen_image, total_loss = sess.run([self.input_image, self.loss])

                    gen_image += self.vgg.mean 

                    run_seconds = time() - start_time 

                    start_time = time()

                    log = '''
                        Iter : %d 
                        Loss : %f
                        Took : %d seconds
                    '''%(idx, total_loss, run_seconds)

                    print log 

                    IMAGE_PREPARE.save_image(self, './output/gen_%d.jpg'%idx, gen_image)




if __name__ == '__main__':
    file_path = './imagenet-vgg-verydeep-19.mat'

    content_image_path = './image/tidy.jpg'

    style_image_path = './image/style0.jpg'

    style_transfer = STYLE_TRANSFER(file_path, style_image_path, content_image_path)

    style_transfer.build()

    style_transfer.train(300)

    print "done."

    
