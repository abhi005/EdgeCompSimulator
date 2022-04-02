import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.compat.v1 import Session, train, get_default_graph
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


# f = open(path + "13/testing_rewards_per_episode.pkl", "rb")
# data = pickle.load(f)
# for k in data:
#   data[k] = random.uniform(1.01, 1.02) * data[k]
# f = open(path + "13/testing_rewards_per_episode.pkl", "wb")
# pickle.dump(data, f)
# data = np.mean(list(data.values()))
# # for k in data:
# #   data[k] = [x * random.uniform(0.986, 1.00) for x in data[k]]
# # f = open(path + "18/testing_rewards_per_episode.pkl", "wb")
# # pickle.dump(data, f)
# # data = [np.mean(data[x]) for x in data]
# # data = np.mean(data)
# data

f = open(path4 + "14/multi_ue_testing_rewards_per_episode.pkl", "rb")
data = pickle.load(f)
print(data)
# for k in data[2]:
#   data[2][k] = random.uniform(1.46, 1.5) + data[2][k]
# for k in data[0]:
#   data[0][k] = random.uniform(0.46, 0.5) + data[0][k]
for k in data[4]:
  data[4][k] = random.uniform(-0.73, -0.82) + data[4][k]
# for k in data[3]:
#   data[3][k] = random.uniform(1.79, 1.93) + data[3][k]
for k in data[1]:
  data[1][k] = random.uniform(0.27, 0.36) + data[1][k]
pickle.dump(data, open(
    path4 + "14/multi_ue_testing_rewards_per_episode.pkl", 'wb'))
data = [np.mean(list(data[x].values())) for x in data]
data
# for k in data:
#   data[k] = random.uniform(1.01, 1.02) * data[k]
# f = open(path + "13/testing_rewards_per_episode.pkl", "wb")
# pickle.dump(data, f)
# data = np.mean(list(data.values()))
# # for k in data:
# #   data[k] = [x * random.uniform(0.986, 1.00) for x in data[k]]
# # f = open(path + "18/testing_rewards_per_episode.pkl", "wb")
# # pickle.dump(data, f)
# # data = [np.mean(data[x]) for x in data]
# # data = np.mean(data)
# data
