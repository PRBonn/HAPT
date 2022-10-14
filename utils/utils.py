import torch
import yaml
import random
import cv2
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image, ImageOps, ImageFilter

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class EmbeddingTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, x):
        return self.transform(x)


class Mix(object):

    def __init__(self, p):
        self.p = p 

    def __call__(self, img):
        if random.random() < self.p:
            return img

        x = np.asarray(img)
        x2 = np.flipud(x)
        x3 = np.fliplr(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                r = random.random()
                if r < 0.33:
                    x[i][j] = x2[i][j]
                elif r < 0.66:
                    x[i][j] = x3[i][j]

        return Image.fromarray(x)

class BackgroundInvariancyAugmentation:
    def __init__(self, p, bg_files):
        self.backgrounds = bg_files
        self.p = p
        self.n_backgrounds = len(bg_files)-1

    def pre_processing(self, image):
        mean = np.mean(image, axis=(0,1))
        broadcast_shape = [1,1]
        broadcast_shape[2-1] = image.shape[2]
        mean = np.reshape(mean, broadcast_shape)
        image = image - mean

        stdDv = np.std(image, axis=(0,1))
        broadcast_shape = [1,1]
        broadcast_shape[2-1] = image.shape[2]
        stdDv = np.reshape(stdDv, broadcast_shape)
        image = image  / (stdDv + 1e-8)

        oMin = 1e-8
        iMin = np.percentile(image, 0.25)

        oMax = 255.0 - 1e-8
        iMax = np.percentile(image, 99.75)

        out = image - iMin
        out *= oMax / (iMax - iMin)
        out[out < oMin] = oMin
        out[out > oMax] = oMax
        return out                    

    def __call__(self, img):

        if random.random() < self.p:
            return img

        img = np.asarray(img)
        img = np.float64(self.pre_processing(img))
        mask = (2*img[:,:,1] - img[:,:,0] - img[:,:,2]) # - (img[:,:,1] - img[:,:,0])
        mask.clip(0,255)

        value = (np.max(mask) - np.min(mask))/3
        mask[ np.where(mask < value) ] = 0.
        mask[ np.where(mask > 0) ] = 255.

        kernel_dilation = np.ones((6,6), np.uint8)
        kernel_erosion = np.ones((3,3), np.uint8)

        mask = cv2.erode(mask, kernel_erosion, iterations=2)
        mask = cv2.dilate(mask, kernel_dilation, iterations=4)

        patch = np.zeros(img.shape, dtype=np.uint8)
        patch[ np.where(mask != 0) ] = img[ np.where(mask != 0) ]
        random_rotation = random.randint(0,180)
        patch = np.asarray(Image.fromarray(patch).rotate(random_rotation))

        new_img = np.asarray( Image.open( "./../data/" + self.backgrounds[ random.randint(0,self.n_backgrounds)  ]) )
        new_img[ np.where( patch!=0) ] = patch[ np.where( patch != 0) ]

        return Image.fromarray(new_img)
       
 
class RandomErasing:
    def __init__(self, p, area):
        self.p = p
        self.area = area

    def __call__(self, img):
        if np.random.random() < self.p:
            return img

        new_img = np.asarray(img)
        S_e = (np.random.random() * self.area + 0.1) * new_img.shape[0] * new_img.shape[1] # random area
        tot = 0

        while tot < S_e:
            y , x = np.random.randint(0, new_img.shape[0]-2) , np.random.randint(0, new_img.shape[1]-2)
            wy, wx = np.random.randint(1, new_img.shape[0] - y) , np.random.randint(1, new_img.shape[1] - x)

            if wy * wx > S_e*2:
                continue

            tot += wy * wx

            random_patch = np.random.rand(wy,wx,3)*255
            new_img[ y : y + wy , x : x + wx , : ] = random_patch
        
        return Image.fromarray(new_img)

class RandomAffine:
    def __init__(self, p, d, t, scale, s, inter):
        self.p = p
        self.degrees = d
        self.translate = t
        self.scale = scale
        self.shear = s
        self.interpolation = inter

    def __call__(self, img):
        if random.random() < self.p:
            return transforms.RandomAffine(degrees = self.degrees,
                                       translate = self.translate,
                                       scale = self.scale,
                                       shear = self.shear,
                                       interpolation = self.interpolation)(img)
        else:
            return img

class Transform:
    def __init__(self):

        # this should have a parameter in cfg passed to it
        with open("./../data/backgrounds.yaml") as f:
            self.bg_files = yaml.safe_load(f)['files']


        self.transform = transforms.Compose([
            
            BackgroundInvariancyAugmentation(0.8, self.bg_files),

            Mix(p=0.9),

            RandomErasing(p=1.0, area = 0.35),
            
            GaussianBlur(p=0.9),
                
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=[ 0 , 0.125])],
                p=1.0,
            ),

            RandomAffine(p = 0.8,
                         d = 180,
                         t = (0.23,0.25),
                         scale = (0.5, 2),
                         s = (0.25 , 0.75 , 0.25 , 0.75 ),
                         inter = transforms.InterpolationMode.BICUBIC),

            transforms.Resize((224,224),interpolation = transforms.InterpolationMode.BICUBIC),
            
            transforms.ToTensor(),
            ])


    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform(x)
        return torch.cat((y1.unsqueeze(0), y2.unsqueeze(0)),0)

class Crop:
    def __init__(self, size):
        self.size = size

    def __call__(self, data_img, global_labels, part_labels):
        px, py, h, w = transforms.RandomCrop.get_params(data_img, self.size)
        img = F.crop(data_img, px, py, h, w)
        global_labels = F.crop(global_labels, px, py, h, w) * 255
        part_labels = F.crop(part_labels, px, py, h, w) * 255
        return {'image': img, 'global_instances':  global_labels, 'parts_instances': part_labels}

class TrainTransform():
    def __init__(self):
        #self.transform_img = transforms.Compose([
            #GaussianBlur(p=0.5),
            #transforms.RandomApply(
            #    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
            #                            saturation=0.2, hue=0.1)],
            #    p=0.5,
            #),
            #])

        self.tensorize = transforms.Compose([
            transforms.ToTensor(),
            ])

        self.crop = Crop((256,512))

    def __call__(self, data):
        #data_img = self.transform_img(data['image'])
        data_final = self.crop( self.tensorize(data['image']), self.tensorize(data['global_instances']).squeeze(), self.tensorize(data['parts_instances']).squeeze())
        return data_final


class ValTransform():
    def __init__(self):
        self.transform_img = transforms.Compose([
            transforms.Resize((256,512), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()])

        self.transform_labels = transforms.Compose([
            transforms.Resize((256,512), transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])


    def __call__(self, data):
        new_data = {}
        new_data['image'] = self.transform_img(data['image'])
        new_data['global_instances'] = 255*self.transform_labels(data['global_instances']).squeeze()
        new_data['parts_instances'] = 255*self.transform_labels(data['parts_instances']).squeeze()
        return new_data


def create_instance_mask(sem, emb, center_pred, center_threshold=0.99, center_sim_threshold=0.8):
    embeddings = emb.squeeze().permute(1,2,0)
    center_predictions = center_pred.squeeze()
    semantic = torch.argmax(torch.softmax(sem.squeeze(), dim=0), dim=0)
    center_confidences = (center_predictions > center_threshold) * semantic
   
    x_centers, y_centers = (center_confidences > 0.).nonzero(as_tuple=True)
    
    while(len(x_centers) == 0 and center_threshold >= 0.5):
        #print("Center threshold", str(center_threshold), "failed. Trying:", str(center_threshold-0.05))
        center_threshold -= 0.05
        center_confidences = (center_predictions > center_threshold) * semantic
        x_centers, y_centers = (center_confidences > 0.).nonzero(as_tuple=True)

    if len(x_centers) == 0: return torch.zeros_like(center_pred)

    centers_embeddings = embeddings[x_centers,y_centers]
    cos_sim_centers_to_centers = centers_embeddings @ centers_embeddings.transpose(1,0)
    
    assignment = cos_sim_centers_to_centers > center_sim_threshold
    assignment_with_cost = assignment * center_predictions[x_centers,y_centers]
    center_indices_no_blob = torch.argmax(assignment_with_cost,dim=1)

    unique_indices = torch.unique(center_indices_no_blob)
    centers_embeddings_no_blob = centers_embeddings[unique_indices, :]

    cos_sim_points_to_centers = embeddings @ centers_embeddings_no_blob.transpose(1,0)

    max_cos, instance_mask = torch.max(cos_sim_points_to_centers, dim=2)
    instance_mask += 1

    return instance_mask * semantic 

def create_instance_mask_distance_based(sem, emb, to_centers_th=0.8):
    # put the embeddings as [H, W, embedding_dim]
    embeddings = emb.permute(1,2,0)
    
    # compute centers, if there are no centers return zero mask
    x_centers, y_centers = (sem > 0.).nonzero(as_tuple=True)
    if len(x_centers) == 0: return torch.zeros_like(sem)

    # extract embeddings for each center
    centers_embeddings = embeddings[x_centers,y_centers]
    # compute the distance between each pair of centers
    dist_centers_to_centers = torch.cdist(centers_embeddings, centers_embeddings, p=2, compute_mode='use_mm_for_euclid_dist') # p-th root
    # assignment is a boolean matrix to suppress each center which is too much similar to another one (same instance)
    assignment = (dist_centers_to_centers < to_centers_th).long()
    #assignment_with_cost = assignment * center_predictions[x_centers,y_centers]
    # here we extract the final centers, should be one pixel for each instance
    center_indices_no_blob = torch.argmax(assignment,dim=1)
    unique_indices = torch.unique(center_indices_no_blob)
    # extract the embeddings of the final centers
    centers_embeddings_no_blob = centers_embeddings[unique_indices, :]
    # compute distance between each pixel and the centers
    dist_points_to_centers = torch.cdist(embeddings,centers_embeddings_no_blob, p=2, compute_mode='use_mm_for_euclid_dist') # p-th root
    # assign each pixel to the center which is closer in the embedding space
    _, instance_mask = torch.min(dist_points_to_centers, dim=2)
    # first element in the assignment is 0, but 0 is the class stuff so we add 1 
    instance_mask += 1
    
    # return the mask filtering out everything which is not in the semantic mask (stuff)
    return instance_mask * sem


