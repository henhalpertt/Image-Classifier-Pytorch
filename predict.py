import argparse
from model import load_model,predict
import numpy as np
import json

# 2 arguments must be entered: <path to the image> and <path to the trained model>

parser = argparse.ArgumentParser(description = "testing")
parser.add_argument('path_image', type=str)
parser.add_argument('path_checkpoint', type=str)
parser.add_argument('--top_k', nargs='?', const=1, type=int, default=1)
parser.add_argument('--category_names', nargs='?', const=1, type=str, default='cat_to_name.json')
parser.add_argument('--gpu',action='store_true')
args = parser.parse_args()

if args.gpu:
    args.gpu = 'cuda'
else:
    args.gpu = 'cpu'

print("path_image: ", args.path_image,
  "path_checkpoint:", args.path_checkpoint,
  "top_k:", args.top_k,
  "category_names:", args.category_names,
  "gpu:", args.gpu
 )

model_checkpt = load_model(args.path_checkpoint)

prob, classes = predict(args.path_image, model_checkpt, args.top_k, args.gpu)
max_probability = prob[np.argmax(prob)]
label = classes[np.argmax(prob)]

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

if args.top_k == 1:
    class_name = cat_to_name[label]
    print(f"Flower class number: {classes}, class name: {class_name} Probability: {np.round(prob,8)}")
else:
    print(f"Flower class labels: {classes}, Probability: {np.round(prob,8)}")
#     print("class names:")
    classes_names = []
    for k in classes:
        classes_names.append(cat_to_name[k])
    print(f"Flower class names: {classes_names}")


 
