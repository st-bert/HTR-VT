import os
import re
import sys
import argparse
from collections import OrderedDict
import random
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from data import dataset
from model import HTR_VT


def preprocess_image(image_paths, img_size):
    image_tensors = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('L')
        transform_fn = transforms.Compose([
            transforms.Resize(tuple(img_size)),
            transforms.ToTensor()
        ])
        image_tensor = transform_fn(image).unsqueeze(0)
        image_tensors.append(image_tensor)
    return torch.cat(image_tensors, dim=0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nb_cls', type=int, default=10)
    parser.add_argument('--img-size', default=[256, 256], type=int, nargs='+')
    parser.add_argument('--pth_path', type=str, default='../output/custom_dataset/best_CER.pth')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--image_folder', type=str, default='./test_images/')
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--max_seq_length', type=int, default=10)

    args = parser.parse_args()

    # randomly select num_images from image_folder
    image_files = os.listdir(args.image_folder)
    image_files = [f for f in image_files if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
    image_files = random.sample(image_files, args.num_images)

    image_paths = [os.path.join(args.image_folder, image_file) for image_file in image_files]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1], max_seq_length=args.max_seq_length)
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

    converter = utils.CrossEntropyConverter(args.max_seq_length)

    image_tensor = preprocess_image(image_paths, args.img_size)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        preds = model(image_tensor)
        preds = preds.float()
        
        # Reshape predictions for cross entropy loss
        batch_size, seq_len, vocab_size = preds.size()
        
        # For decoding predictions to text
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        
        # Get softmax probabilities for decoding
        preds_softmax = preds.log_softmax(2)
        
        # Get predicted indices
        _, preds_index = preds_softmax.max(2)
        preds_str = converter.decode(preds_index.data, preds_size.data)
        
        # Now preds_str contains predictions for all images
        recognized_texts = preds_str

    for i, image_file in enumerate(image_files):
        print(f"Image: {image_file}, Predicted Text: {recognized_texts[i]}")


if __name__ == '__main__':
    main()
