import pickle

path = 'dataset/mini-imagenet/compacted_datasets/mini_imagenet_val.pickle'

f = open(path, 'rb')
data = pickle.load(f)
print(data)
print(len(data))
f.close()
