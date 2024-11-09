import sys

sys.path.append("pytorch-retinanet")
import torch
from retinanet import model


# produce a fake parser to pass to the class with the options we need
class Parser:
    pass


# creates a parser
parser = Parser()
parser.supervised = True
parser.depth = 50
parser.num_classes = 1
parser.dim_out = 32
parser.model_path = "/Volumes/EX_DRIVE/new_git/weights/best_anomaly.pt"

retinanet = model.resnet50(num_classes=1, pretrained=True)

use_gpu = True

if use_gpu:
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()

if torch.cuda.is_available():
    print('loading to cuda')
    retinanet.load_state_dict(torch.load(parser.model_path))
    retinanet = torch.nn.DataParallel(retinanet).cuda()
else:
    print('loading to cpu')
    dictionary_weights = torch.load(parser.model_path, map_location=torch.device('cpu'))
    retinanet.load_state_dict(dictionary_weights, strict=False)
    retinanet = torch.nn.DataParallel(retinanet)
    print('done loading model')

retinanet.training = False
retinanet.module.freeze_bn()
retinanet.eval();
