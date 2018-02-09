import tensorflow as tf
from tensorflow.core.framework import graph_pb2
import numpy as np
try:
    from src.model import preprocess
except Exception: #ImportError
    import preprocess

import os
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

image_dir = os.path.join("src", "images")
graph_dir = os.path.join("src", "model", "inception_v3_2016_08_28_frozen.pb")
label_dir = os.path.join("src", "model", "labels.txt")

class model(object):
    
    def __init__(self):
        self._graph = None
        self._labels = None
        self._session = None

    def load_graph(self, graph_dir=graph_dir, label_dir=label_dir):
        graph_def = graph_pb2.GraphDef()
        with open(label_dir, 'r') as f:
            self._labels = f.read().split('\n')
        with open(graph_dir, "rb") as f:
            graph_def.ParseFromString(f.read())
        self._graph = tf.import_graph_def(graph_def)
        self._input_op = tf.get_default_graph().get_tensor_by_name('import/input:0')
        self._output_op = tf.get_default_graph().get_tensor_by_name('import/InceptionV3/Predictions/Reshape_1:0')

    def start_session(self):
        self._session = tf.Session()
    
    def close_session(self):
        self._session.close()

    def run(self, filename, new_size):
        if self._session == None:
            self.start_session()
        img = preprocess.preprocess(filename, new_size)
        return self._session.run([self._output_op], feed_dict={self._input_op: [img]})

    def get_label(self, filename, new_size):
        probs = self.run(filename, new_size)
        return self._labels[np.argmax(probs)], np.max(probs)

    def save_graph(self):
        if self._input_op is not None and self._output_op is not None:
            export_path = os.path.join("src", "model", "export", "1") #model version is 1
            print("exporting model to " + export_path)
            builder = saved_model_builder.SavedModelBuilder(export_path)
            tensor_info_x = utils.build_tensor_info(self._input_op)
            tensor_info_y = utils.build_tensor_info(self._output_op)
            prediction_signature = signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_x},
                outputs={'scores': tensor_info_y},
                method_name=signature_constants.PREDICT_METHOD_NAME)
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                tf.Session(), [tag_constants.SERVING],
                signature_def_map={
                    'predict_images':prediction_signature,},
                    legacy_init_op=legacy_init_op)
            builder.save()

"""
All tests must be written as started from CNNV3 directory
"""
if __name__ == "__main__":
    m = model()
    m.load_graph()
    new_size = [299, 299]
    m.start_session()
    for f in os.listdir(image_dir):
        ff = os.path.abspath(os.path.join(image_dir, f))
        print(ff)
        print(m.get_label(ff, new_size))
    m.close_session()
    m.save_graph()
    print("Graph Saved via Save Builder.")