import torch
import yaml
import torchvision
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import os
import os.path as path
from PIL import Image, ImageFile
import utils.utils as utils
import numpy as np
import torch.nn.functional as F
import glob

ImageFile.LOAD_TRUNCATED_IMAGES = True

class StatDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        # from cfg i can access to all my shit
        # as data path, data size and so on 
        self.cfg = cfg
        self.len = -1
        self.setup()
        self.loader = [ self.train_dataloader(), self.val_dataloader() ]

    def prepare_data(self):
        # Augmentations are applied using self.transform 
        # no data to download, for now everything is local 
        pass

    def setup(self, stage=None):

        self.mode = self.cfg['train']['mode']

        if stage == 'fit' or stage is None:
            self.data_train = SugarBeets(self.cfg['data']['ft-path'], 'train', overfit=self.cfg['train']['overfit'])
            self.data_val = SugarBeets(self.cfg['data']['ft-path'], 'val', overfit=self.cfg['train']['overfit'])
            # self.data_train = GrowliFlower(self.cfg['data']['ft-path'], 'Train', overfit=self.cfg['train']['overfit'])
            # self.data_val = GrowliFlower(self.cfg['data']['ft-path'], 'Val', overfit=self.cfg['train']['overfit'])
        return

    def train_dataloader(self):
        loader = DataLoader(self.data_train, 
                            batch_size = self.cfg['train']['batch_size'] // self.cfg['train']['n_gpus'],
                            num_workers = self.cfg['train']['workers'],
                            pin_memory=True,
                            shuffle=True)
        self.len = self.data_train.__len__()
        return loader

    def val_dataloader(self):
        if self.mode == 'pt': pass
        loader = DataLoader(self.data_val, 
                            batch_size = 1, #self.cfg['train']['batch_size'] // self.cfg['train']['n_gpus'],
                            num_workers = self.cfg['train']['workers'],
                            pin_memory=True,
                            shuffle=False)
        self.len = self.data_val.__len__()
        return loader
            
    def test_dataloader(self):
        pass

#################################################
#################### Datasets ###################
#################################################

""" BOSCH DATASET """

class SugarBeets(Dataset):
    def __init__(self, datapath, mode, overfit=False):
        super().__init__()
        
        self.datapath = datapath
        self.mode = mode
        self.overfit = overfit 

        if self.overfit:
            self.datapath += '/images/train'
        else:
            self.datapath += '/images/' + mode

        self.all_imgs = [os.path.join(self.datapath,x) for x in os.listdir(self.datapath) if ".png" in x]
        self.all_imgs.sort()

        global_annotations_path = os.path.join(self.datapath.replace('images','annos'), 'global')
        parts_annotations_path = os.path.join(self.datapath.replace('images','annos'), 'parts')
        self.global_instance_list =[os.path.join(global_annotations_path,x) for x in os.listdir(global_annotations_path) if ".semantic" in x]
        self.parts_instance_list =[os.path.join(parts_annotations_path,x) for x in os.listdir(parts_annotations_path) if ".semantic" in x]
        self.global_instance_list.sort()
        self.parts_instance_list.sort()
            
        self.transform = utils.ValTransform() 
        self.len = len(self.all_imgs)
      

    def get_centers(self, mask):
        if mask.sum() == 0:
            return torch.zeros((0, 4), device=mask.device, dtype=torch.float)
        
        masks = F.one_hot(mask.long())
        masks = masks.permute(2,0,1)[1:,:,:]
        num, H, W = masks.shape

        center_mask = torch.zeros( (H, W) , device=masks.device, dtype=torch.float)

        for submask in masks:
            if submask.sum() == 0:
                continue
            x, y = torch.where(submask != 0)
            xy = torch.cat([x.unsqueeze(0),y.unsqueeze(0)], dim=0)
            mu, _ = torch.median(xy,dim=1, keepdim=True)
            center_idx = torch.argmin(torch.sum(torch.abs(xy - mu), dim=0))
            center = xy[:,center_idx]
            center_mask[center[0], center[1]] = 1.
    
        return center_mask

    def get_offsets(self, mask, centers):
        if mask.sum() == 0:
            return torch.zeros((0, 4), device=mask.device, dtype=torch.float)

        masks = F.one_hot(mask.long())
        masks = masks.permute(2,0,1)[1:,:,:]
        num, H, W = masks.shape
        
        total_mask = torch.zeros((H,W,2), device = masks.device, dtype=torch.float)

        for submask in masks:
            coords = torch.ones((H,W,2))
            tmp = torch.ones((H,W,2))
            tmp[:,:,1] = torch.cumsum(coords[:,:,0],0) - 1
            tmp[:,:,0] = torch.cumsum(coords[:,:,1],1) - 1

            current_center = torch.where(submask * centers)
            
            offset_mask = (tmp - torch.tensor([current_center[1], current_center[0]])) * submask.unsqueeze(2)
            total_mask += offset_mask
        
        return total_mask


    def __getitem__(self, index):
        
        if self.mode == 'pt': 
            img_loc = os.path.join(self.datapath, self.all_imgs[index])
            img = Image.open(img_loc).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor

        # TRAINING MODE
        # load image
        sample = {}

        image = Image.open(self.all_imgs[index])
        width, height = image.size
        sample['image'] = image

        global_annos = np.fromfile(self.global_instance_list[index], dtype=np.uint32)
        global_annos = global_annos.reshape(height, width)

        parts_annos = np.fromfile(self.parts_instance_list[index], dtype=np.uint32)
        parts_annos = parts_annos.reshape(height, width)

        global_instances = global_annos >> 16 # get upper 16-bits

        # instance ids might start at high numbers and are not successive, thus we make sure that this is the case
        global_instance_ids = np.unique(global_instances)[1:] # no background
        global_instances_successive =  np.zeros_like(global_instances)
        for idx, id_ in enumerate(global_instance_ids):
            instance_mask = global_instances == id_
            global_instances_successive[instance_mask] = idx + 1
        global_instances = global_instances_successive

        assert np.max(global_instances) <= 255, 'Currently we do not suppot more than 255 instances in an image'

        parts_instances = parts_annos >> 16 # get upper 16-bits
        # instance ids might start at high numbers and are not successive, thus we make sure that this is the case
        parts_instance_ids = np.unique(parts_instances)[1:] # no background
        parts_instances_successive =  np.zeros_like(parts_instances)
        for idx, id_ in enumerate(parts_instance_ids):
            instance_mask = parts_instances == id_
            parts_instances_successive[instance_mask] = idx + 1
        parts_instances = parts_instances_successive

        assert np.max(parts_instances) <= 255, 'Currently we do not suppot more than 255 instances in an image'

        global_instances = Image.fromarray(np.uint8(global_instances))
        parts_instances = Image.fromarray(np.uint8(parts_instances))

        sample['global_instances'] = global_instances
        sample['parts_instances'] = parts_instances
       
        # transform
        if(self.transform is not None):
            sample = self.transform(sample)
         

        semantic = sample['global_instances'].bool().long()
        sample['global_centers'] = self.get_centers(sample['global_instances'])
        center_masks = F.one_hot((sample['global_centers'] * sample['global_instances']).long())[:,:,1:]
        center_masks = torchvision.transforms.GaussianBlur(11, 5.0)(center_masks.permute(2,0,1).unsqueeze(0).float()).squeeze()
        sample['global_offsets'] = self.get_offsets(sample['global_instances'], sample['global_centers']).permute(2,0,1)
        sample['global_centers'] = (torch.max(center_masks, dim=0)[0] / torch.max(center_masks)) * semantic

        sample['parts_centers'] = self.get_centers(sample['parts_instances'])
        sample['parts_offsets'] = self.get_offsets(sample['parts_instances'], sample['parts_centers']).permute(2,0,1)
        center_masks = F.one_hot((sample['parts_centers'] * sample['parts_instances']).long())[:,:,1:]
        center_masks = torchvision.transforms.GaussianBlur(7, 3.0)(center_masks.permute(2,0,1).unsqueeze(0).float()).squeeze()
        sample['parts_centers'] = (torch.max(center_masks, dim=0)[0] / torch.max(center_masks)) * semantic

        sample['loss_masking'] = torch.ones(sample['global_instances'].shape)    # 1 where loss has to be computed, 0 elsewhere

        return sample
       

    def __len__(self):
        if self.overfit: 
            return self.overfit
        return self.len


""" GROWLIFLOWER DATASET """

class GrowliFlower(Dataset):
    def __init__(self, datapath, mode, overfit=False):
        super().__init__()
        
        self.datapath = datapath
        self.mode = mode
        self.overfit = overfit 

        if self.overfit:
            self.datapath += '/images/Train'
        else:
            self.datapath += '/images/' + mode

        
        self.all_imgs = [os.path.join(self.datapath,x) for x in os.listdir(self.datapath) if ".jpg" in x]
        self.all_imgs.sort()

        global_annotations_path = os.path.join(self.datapath.replace('images','labels'), 'maskPlants')
        parts_annotations_path = os.path.join(self.datapath.replace('images','labels'), 'maskLeaves')
        void_annotations_path = os.path.join(self.datapath.replace('images','labels'), 'maskVoid')
        
        self.global_instance_list =[os.path.join(global_annotations_path,x) for x in os.listdir(global_annotations_path)]
        self.parts_instance_list =[os.path.join(parts_annotations_path,x) for x in os.listdir(parts_annotations_path)]
        self.void_instance_list =[os.path.join(void_annotations_path,x) for x in os.listdir(void_annotations_path)]

        self.global_instance_list.sort()
        self.parts_instance_list.sort()
        self.void_instance_list.sort()

        self.len = len(self.all_imgs)

        self.transform = torchvision.transforms.ToTensor()
      

    def __getitem__(self, index):
        sample = {}

        image = Image.open(self.all_imgs[index])
        width, height = image.size
        sample['image'] = self.transform(image).squeeze()

        global_annos = np.array(Image.open(self.global_instance_list[index])).astype(np.int32)
        parts_annos = np.array(Image.open(self.parts_instance_list[index])).astype(np.int32)
        void_annos = np.array(Image.open(self.void_instance_list[index])).astype(np.int32)

        sample['global_instances'] = np.logical_not(void_annos) * global_annos 
        sample['parts_instances'] = parts_annos

        # instance ids might start at high numbers and are not successive, thus we make sure that this is the case
        global_instance_ids = np.unique(sample['global_instances'])[1:] # no background
        global_instances_successive =  np.zeros_like(sample['global_instances'])
        for idx, id_ in enumerate(global_instance_ids):
            instance_mask = sample['global_instances'] == id_
            global_instances_successive[instance_mask] = idx + 1
        global_instances = global_instances_successive

        assert np.max(global_instances) <= 255, 'Currently we do not support more than 255 instances in an image'

        # instance ids might start at high numbers and are not successive, thus we make sure that this is the case
        parts_instance_ids = np.unique(sample['parts_instances'])[1:] # no background
        parts_instances_successive =  np.zeros_like(sample['parts_instances'])
        for idx, id_ in enumerate(parts_instance_ids):
            instance_mask = sample['parts_instances'] == id_
            parts_instances_successive[instance_mask] = idx + 1
        parts_instances = parts_instances_successive

        assert np.max(parts_instances) <= 255, 'Currently we do not support more than 255 instances in an image'

        sample['global_instances'] = self.transform(global_instances).squeeze()
        sample['parts_instances'] = self.transform(parts_instances).squeeze()
        
        semantic = sample['global_instances'].bool().long()
        sample['global_centers'] = self.get_centers(sample['global_instances'])

        if sample['global_centers'].sum() != 0:
            center_masks = F.one_hot((sample['global_centers'] * sample['global_instances']).long())[:,:,1:]
            center_masks = torchvision.transforms.GaussianBlur(11, 5.0)(center_masks.permute(2,0,1).unsqueeze(0).float()).squeeze()
            if len(center_masks.shape) < 3: center_masks = center_masks.unsqueeze(0)
            sample['global_offsets'] = self.get_offsets(sample['global_instances'], sample['global_centers']).permute(2,0,1)
            sample['global_centers'] = (torch.max(center_masks, dim=0)[0] / torch.max(center_masks)) * semantic
        else:
            sample['global_offsets'] = torch.zeros((sample['global_instances'].shape[-2], sample['global_instances'].shape[-1], 2), device=sample['global_instances'].device, dtype=torch.float).permute(2,0,1)
            sample['global_centers'] = torch.zeros(sample['global_instances'].shape, device=sample['global_instances'].device, dtype=torch.float)

        sample['parts_centers'] = self.get_centers(sample['parts_instances'])
        if sample['parts_centers'].sum() != 0:
            center_masks = F.one_hot((sample['parts_centers'] * sample['parts_instances']).long())[:,:,1:]
            center_masks = torchvision.transforms.GaussianBlur(7, 3.0)(center_masks.permute(2,0,1).unsqueeze(0).float()).squeeze()
            if len(center_masks.shape) < 3: center_masks = center_masks.unsqueeze(0)
            sample['parts_offsets'] = self.get_offsets(sample['parts_instances'], sample['parts_centers']).permute(2,0,1)
            sample['parts_centers'] = (torch.max(center_masks, dim=0)[0] / torch.max(center_masks)) * semantic
        else:
            sample['parts_offsets'] = torch.zeros((sample['parts_instances'].shape[-2], sample['parts_instances'].shape[-1], 2), device=sample['parts_instances'].device, dtype=torch.float).permute(2,0,1)
            sample['parts_centers'] = torch.zeros(sample['parts_instances'].shape, device=sample['parts_instances'].device, dtype=torch.float)

        sample['loss_masking'] = self.transform(np.logical_not(void_annos)).squeeze().long()    # 1 where loss has to be computed, 0 elsewhere

        return sample

    def get_centers(self, mask):
        if mask.sum() == 0:
            # return torch.zeros((0, 4), device=mask.device, dtype=torch.float)
            return torch.zeros(mask.shape, device=mask.device, dtype=torch.float)
        
        masks = F.one_hot(mask.long())
        masks = masks.permute(2,0,1)[1:,:,:]
        num, H, W = masks.shape
        center_mask = torch.zeros( (H, W) , device=masks.device, dtype=torch.float)

        for submask in masks:
            if submask.sum() == 0:
                continue
            x, y = torch.where(submask != 0)
            xy = torch.cat([x.unsqueeze(0),y.unsqueeze(0)], dim=0)
            mu, _ = torch.median(xy,dim=1, keepdim=True)
            center_idx = torch.argmin(torch.sum(torch.abs(xy - mu), dim=0))
            center = xy[:,center_idx]
            center_mask[center[0], center[1]] = 1.
    
        return center_mask

    def get_offsets(self, mask, centers):
        if mask.sum() == 0:
            return torch.zeros((0, 4), device=mask.device, dtype=torch.float)

        masks = F.one_hot(mask.long())
        masks = masks.permute(2,0,1)[1:,:,:]
        num, H, W = masks.shape
        
        total_mask = torch.zeros((H,W,2), device = masks.device, dtype=torch.float)

        for submask in masks:
            coords = torch.ones((H,W,2))
            tmp = torch.ones((H,W,2))
            tmp[:,:,1] = torch.cumsum(coords[:,:,0],0) - 1
            tmp[:,:,0] = torch.cumsum(coords[:,:,1],1) - 1

            current_center = torch.where(submask * centers)
            
            offset_mask = (tmp - torch.tensor([current_center[1], current_center[0]])) * submask.unsqueeze(2)
            total_mask += offset_mask
        
        return total_mask
       

    def __len__(self):
        if self.overfit: 
            return self.overfit
        return self.len
