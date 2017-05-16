class Inceptionv3:
    tensor_name_input_jpeg = "DecodeJpeg/contents:0"
    tensor_name_input_image = "DecodeJpeg:0"
    tensor_name_resized_image = "ResizeBilinear:0"
    tensor_name_softmax = "softmax:0"
    tensor_name_softmax_logits = "softmax/logits:0"
    tensor_name_transfer_layer = "pool_3:0"
    tensor_name_transfer_conv_layer = 'mixed_10/join/concat_dim:0'
    tensor_name_transfer_convJoin_layer = "mixed_10/join:0"
    tensor_name_transfer_convMix_layer="mixed_10/tower_2/conv:0"

    def __init__(self , path_graph_def):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.gfile.FastGFile(path_graph_def, 'rb') as file:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(file.read())
                tf.import_graph_def(graph_def, name='')
        self.y_pred             = self.graph.get_tensor_by_name(self.tensor_name_softmax)
        self.y_logits           = self.graph.get_tensor_by_name(self.tensor_name_softmax_logits)
        self.resized_image      = self.graph.get_tensor_by_name(self.tensor_name_resized_image)
        self.transfer_layer     = self.graph.get_tensor_by_name(self.tensor_name_transfer_layer)
        self.transfer_conv_layer = self.graph.get_tensor_by_name(self.tensor_name_transfer_conv_layer)
        self.transfer_convJoin_layer = self.graph.get_tensor_by_name(self.tensor_name_transfer_convJoin_layer)
        self.transfer_convMix_layer=self.graph.get_tensor_by_name(tensor_name_transfer_convMix_layer)
        self.transfer_len = self.transfer_layer.get_shape()[3]
        self.transfer_conv_len = self.transfer_conv_layer.get_shape()
        
        self.session = tf.Session(graph=self.graph)
        
    def show_all_op(self):
        all_ops=self.session.graph.get_operations()
        for ele in all_ops:
            print ele.name
    def close(self):
        self.session.close()
    def _write_summary(self, logdir='summary/'):
        writer = tf.train.SummaryWriter(logdir=logdir, graph=self.graph)
        writer.close()
    def _create_feed_dict(self, image_path=None, image=None):
        if image is not None:
            feed_dict = {self.tensor_name_input_image: image}
        elif image_path is not None:
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            feed_dict = {self.tensor_name_input_jpeg: image_data}
        else:
            raise ValueError("Either image or image_path must be set.")
        return feed_dict

    def classify(self, image_path=None, image=None):
        feed_dict = self._create_feed_dict(image_path=image_path, image=image)
        pred = self.session.run(self.y_pred, feed_dict=feed_dict)
        pred = np.squeeze(pred)
        return pred

    def get_resized_image(self, image_path=None, image=None):
        feed_dict = self._create_feed_dict(image_path=image_path, image=image)
        resized_image = self.session.run(self.resized_image, feed_dict=feed_dict)
        resized_image = resized_image.squeeze(axis=0)
        resized_image = resized_image.astype(float) / 255.0
        return resized_image

    def print_scores(self, pred, k=10, only_first_name=True):
        print 'print_score'
        idx = pred.argsort()
        print idx
        top_k = idx[-k:]
        print top_k
        for cls in reversed(top_k):
            name = self.name_lookup.cls_to_name(cls=cls, only_first_name=only_first_name)
            score = pred[cls]
            print name , score
            print("{0:>6.2%} : {1}".format(score, name))
    
    def transfer_values(self, transfer_layer , image_path=None, image=None):
        feed_dict = self._create_feed_dict(image_path=image_path, image=image)
        transfer_values = self.session.run(self.transfer_layer , feed_dict=feed_dict)
        #transfer_values = np.squeeze(transfer_values)
        return transfer_values
def process_images(fn, images=None, image_paths=None):
        using_images = images is not None
        if using_images:
            num_images = len(images)
        else:
            num_images = len(image_paths)
        result = [None] * num_images
        for i in range(num_images):
            msg = "\r- Processing image: {0:>6} / {1}".format(i+1, num_images)
            sys.stdout.write(msg)
            sys.stdout.flush()
            if using_images:
                result[i] = fn(image=images[i])
            else:
                result[i] = fn(image_path=image_paths[i])
        print()
        result = np.array(result)
        return result
    
def transfer_values_cache(cache_path, model, images=None, image_paths=None):
        def fn():
            return process_images(fn=model.transfer_values, images=images, image_paths=image_paths)
        transfer_values = cache(cache_path=cache_path, fn=fn)
        return transfer_values
def transfer_conv_values_cache(cache_path, model, images=None, image_paths=None):
        def fn():
            return process_images(fn=model.transfer_conv_values, images=images, image_paths=image_paths)
        transfer_values = cache(cache_path=cache_path, fn=fn)
        return transfer_values
def transfer_convJoin_values_cache(cache_path, model, images=None, image_paths=None):
    def fn():
        return process_images(fn=model.transfer_convJoin_values, images=images, image_paths=image_paths)
    transfer_values = cache(cache_path=cache_path, fn=fn)
    return transfer_values
def transfer_convMix_values_cache(cache_path, model, images=None, image_paths=None):
    def fn():
        return process_images(fn=model.transfer_convMix_values, images=images, image_paths=image_paths)
    transfer_values = cache(cache_path=cache_path, fn=fn)
    return transfer_values
    