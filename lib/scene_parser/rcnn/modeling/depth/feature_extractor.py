import torch
import  torch.nn as nn #
from torchvision.transforms import functional as F
import os,sys
print(sys.path)
from layers.roi_align import ROIAlign
from PIL import Image
import numpy as np
import pickle

class ROIFeatureExtractor(nn.Module):
    def __init__(self):
        super(ROIFeatureExtractor, self).__init__()
        self.conv = nn.Conv2d(3,3,3,1,1)
        self.roi_align = ROIAlign(
            output_size    = (7, 7),
            spatial_scale  = 1,
            sampling_ratio = 1    
        )
        
    def forward(self, x, proposals):
        x = self.conv(x)
        out = self.roi_align(x, proposals)
        return out
    
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fe = ROIFeatureExtractor()
    image = Image.open('/home/zduanmu/360videodata/data/NYUv2/image/train/00867.png')
    img = np.array(image)
    x = F.to_tensor(img).float()
    # x = x.permute([2,0,1])
    x = x.unsqueeze(0) 
       
    relation = open('/home/zduanmu/360videodata/data/NYUv2/image/train/00867.pkl','rb')
    relation_features = pickle.load(relation)
    detections      = relation_features['detections']
    proposals = detections['bbox'][:20]
    proposals = np.hstack([proposals, np.zeros([proposals.shape[0],1])])
    proposals = F.to_tensor(proposals).float()
    # proposals  = proposals.squeeze()
    
    x = x.to(device)
    proposals = proposals.to(device)
    fe.to(device)
    with torch.no_grad():
        y = fe(x,proposals)
        
    y = y.cpu()
    print(y.shape)