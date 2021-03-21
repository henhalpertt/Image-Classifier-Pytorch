from model import model_ft,save_model
import argparse
"""
train a new network on a dataset and save the model as a checkpoint
data directory must be given
Use vgg13 or alexnet pretrained models

Input format:
python train.py data_dir --learning_rate <float> --hidden_units <int> --epochs <int>

example:
python train.py flowers --learning_rate 0.01 --hidden_units 512 --epochs 20

For the person who tests:
# alexnet produced ~85% validation accuracy in 24minutes. vgg13 produced ~88% validation accuracy in 47minutes

"""
parser = argparse.ArgumentParser(description = "pass data_directory")
parser.add_argument('data_directory', type=str)
parser.add_argument('--save_dir', nargs='?', const=1, type=str, default='checkpoint_alexnet2.pth')
parser.add_argument('--arch', nargs='?', const=1, type=str, default='alexnet')
parser.add_argument('--learning_rate', nargs='?', const=1, type=float, default=0.001)
parser.add_argument('--epochs', nargs='?', const=1, type=int, default=1)
parser.add_argument('--hidden_units', nargs='?', const=1, type=int, default=512)
parser.add_argument('--gpu',action='store_true')
args = parser.parse_args()

if args.gpu:
    args.gpu = 'cuda'
else:
    args.gpu = 'cpu'

print(f"data_directory: {args.data_directory}, save_dir: {args.save_dir}, arch: {args.arch}, learning_rate: {args.learning_rate}, epochs: {args.epochs}, hidden_units: {args.hidden_units}, device: {args.gpu}")

model = model_ft(args.data_directory, save_dir=args.save_dir,
                 architecture=args.arch, lr=args.learning_rate,
                 hidden_units=args.hidden_units, epochs=args.epochs, device=args.gpu)
