import argparse
from model import load_model

# 2 arguments must be entered: <path to the image> and <path to the trained model>

parser = argparse.ArgumentParser(description = "testing")
parser.add_argument('path_image', type=str)
parser.add_argument('path_checkpoint', type=str)
parser.add_argument('--top_k', nargs='?', const=1, type=int, default=3)
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
print(model_checkpt)
