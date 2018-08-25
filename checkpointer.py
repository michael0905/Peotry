import os
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
model_dir = './experiments/PeotryGRU/checkpoint'
checkpoint_path = os.path.join(model_dir, "-812")

print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=False)
