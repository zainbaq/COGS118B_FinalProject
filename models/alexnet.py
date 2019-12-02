import torch
from PIL import Image
from torchvision import transforms

from train_test_split import train_test_split
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

data_dir = 'datasets/final_data.csv'

df = pd.read_csv(data_dir, sep=',')

tr_inp, tr_labels, te_inp, te_labels = train_test_split(df)

model = torch.hub.load(
    'pytorch/vision:v0.4.2', 
    'alexnet',
    pretrained=True
)

model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        # Normalization as described by AlexNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.2225]
    )
])

def to_rgb(im):
    im = np.reshape(im.to_numpy(), (256, 256))
    print(im.size)
    im_rgb = Image.new("RGB", im.shape)
    im_rgb.paste(im)
    return im_rgb

in_tensor = preprocess(to_rgb(tr_inp.iloc[0]))
in_batch = in_tensor.unsqueeze(0)

if torch.cuda.is_available():
    in_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(in_batch)

print(output[0])
print(torch.nn.functional.softmax(output[0], dim=0))

# imarray = np.array(im_rgb)
# # im_0 = tr_inp.iloc[0].to_numpy()
# # im_0 = np.reshape(im_0, (256, 256))
# plt.imshow(imarray)
# plt.show()

