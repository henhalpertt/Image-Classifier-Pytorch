from model import model_ft,save_model
import argparse

# if no arguments are given, arguments will default to specific values
# data directory must be given

parser = argparse.ArgumentParser(description = "testing")
parser.add_argument('data_directory', type=str)
parser.add_argument('--save_dir', nargs='?', const=1, type=str, default='checkpoint.pth')
parser.add_argument('--arch', nargs='?', const=1, type=str, default='vgg13')
parser.add_argument('--learning_rate', nargs='?', const=1, type=float, default=0.001)
parser.add_argument('--epochs', nargs='?', const=1, type=int, default=1)
parser.add_argument('--hidden_units', nargs='?', const=1, type=int, default=512)
parser.add_argument('--gpu',action='store_true')
args = parser.parse_args()

if args.gpu:
    args.gpu = 'cuda'
else:
    args.gpu = 'cpu'
print("--------------")
print("data_directory: ", args.data_directory,
      "save_dir:", args.save_dir,
      "arch:", args.arch,
      "learning_rate:", args.learning_rate,
      "epochs:", args.epochs,
      "hidden_units:", args.hidden_units,
      "gpu:", args.gpu
     )

model = model_ft(args.data_directory, save_dir=args.save_dir,
                 architecture=args.arch, lr=args.learning_rate,
                 hidden_units=args.hidden_units, epochs=args.epochs, device=args.gpu)

if args.save_dir:
    save_model(model, args.save_dir)
else:
    print("save_dir not specified")
          
