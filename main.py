import argparse
import os
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import cv2

from unet import ResNet
import matplotlib.pyplot as plt
import time
import importAndProcess as iap

start_cpu_time = time.process_time()

parser = argparse.ArgumentParser()
args = parser.parse_args()
normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
model = ResNet(out_filters=3)
resize_dim = (224, 224)
transforms = Compose([Resize(resize_dim),ToTensor(),normalize])

dataset = iap.lungSegmentDataset(
    os.path.join("data/before.png"),
    os.path.join("data/right-side.png"),
    os.path.join("data/left-side.png"),
    imagetransform=transforms,
    labeltransform=Compose([Resize((264, 264)),ToTensor()]),
    convert_to='RGB',
)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
image = cv2.FILE_STORAGE_FORMAT_MASK
model = torch.nn.DataParallel(model)
show = iap.visualize(dataset)
output_image_sizes = []
with torch.no_grad():
     for i, sample in enumerate(dataloader):
         img = torch.autograd.Variable(sample['image'])
         mask = model(img)
         if not args.non_montgomery:
             image_path=(show.ImageWithGround(i,True,True,save=True))
         image_path=(show.ImageWithMask(i, sample['filename'][0], mask.squeeze().cpu().numpy(), True, True, save=True))


image = cv2.imread(image_path)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
total_storage_used = 0
for root, dirs, files in os.walk('.'):
    for file in files:
        total_storage_used += os.path.getsize(os.path.join(root, file))

print("Total storage used : %s bytes" % total_storage_used)


print("--- CPU Time: %s seconds ---" % (time.process_time() - start_cpu_time))