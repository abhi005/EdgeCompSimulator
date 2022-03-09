import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.compat.v1 import variable_scope, Session, placeholder, layers, distributions, train, global_variables_initializer, train, get_default_graph, get_variable
from os.path import exists
tf.compat.v1.disable_eager_execution()

model_path = "../models/actor_critic_max_10_task.ckpt"
if exists(model_path):
    print("file doesn't exists")
else:
    sess = Session()
    saver = train.import_meta_graph(model_path + ".meta")
    saver.restore(sess, model_path)
    graph = get_default_graph()
    print(graph.get_all_collection_keys())
    print(graph.get_collection(name="trainable_variables"))
    print(graph.get_collection(name="train_op"))
    print(graph.get_tensor_by_name("l1/bias"))
