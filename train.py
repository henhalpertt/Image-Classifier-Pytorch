from model import model_ft
import argparse

parser = argparse.ArgumentParser(description = "testing")
parser.add_argument('data_directory', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--arch', type=str)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--epochs', type=int)
parser.add_argument('--hidden_units', type=int, default=1)
parser.add_argument('--gpu',action='store_true')
args = parser.parse_args()
# train_parser.add_argument('--processor', action="store", dest="processor", default="GPU")
# print("Image Directory: ", args.data_directory)

if args.gpu:
    args.gpu = 'cuda'
else:
    args.gpu = 'cpu'
print("cpu: ", args.gpu, "data_dir:", args.data_directory, "save_dir:", args.save_dir)


# model = model_ft(args.data_directory, save_dir=args.save_dir,
#                  architecture=args.arch, lr=args.learning_rate,
#                  hidden_units=args.hidden_units, epochs=args.epochs, device=args.gpu)




