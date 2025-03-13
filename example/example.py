import os
import re
import sys
import argparse
from collections import OrderedDict

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from data import dataset
from model import HTR_VT


def preprocess_image(image_path, img_size):
    image = Image.open(image_path).convert('L')
    transform_fn = transforms.Compose([
        transforms.Resize(tuple(img_size)),
        transforms.ToTensor()
    ])
    image_tensor = transform_fn(image).unsqueeze(0)
    return image_tensor


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nb_cls', type=int, default=11)
    parser.add_argument('--img-size', default=[256, 256], type=int, nargs='+')
    parser.add_argument('--data_path', type=str, default='../data/custom_dataset/lines/')
    parser.add_argument('--pth_path', type=str, default='../output/custom_dataset/best_CER.pth')
    parser.add_argument('--train_data_list', type=str, default='../data/custom_dataset/train.ln')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--image_path', type=str, default='./valid_1.jpeg')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1])
    ckpt = torch.load(args.pth_path, map_location='cpu')

    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k, v in ckpt['state_dict_ema'].items():
        if re.search(pattern, k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v

    model.load_state_dict(model_dict, strict=True)
    model = model.to(device)
    model.eval()

    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)
    converter = utils.CTCLabelConverter()

    image_tensor = preprocess_image(args.image_path, args.img_size)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        preds = model(image_tensor)
        preds = preds.float()
        preds_size = torch.IntTensor([preds.size(1)])
        preds = preds.permute(1, 0, 2).log_softmax(2)
        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        preds_str = converter.decode(preds_index.data, preds_size.data)
        recognized_text = preds_str[0]

    print(f"Recognized_text: {recognized_text}")


if __name__ == '__main__':
    main()
