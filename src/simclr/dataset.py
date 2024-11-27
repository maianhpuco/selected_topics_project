import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets
import pandas as pd
from PIL import Image
from skimage import io, img_as_ubyte

np.random.seed(0)
import cv2
import numpy as np

np.random.seed(0)



import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets
import pandas as pd
from PIL import Image
from skimage import io, img_as_ubyte

np.random.seed(0)
import cv2
import numpy as np

np.random.seed(0)

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
#            print(self.kernel_size)
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

class SimCLRDataloader(object):
    def __init__(self, batch_size=32, num_workers=1, input_shape='(224, 224, 3)', s=1, csv_path=None, is_train=True):
        '''
        csv_train =. './selected_topics_project/data/csv_files/train.csv' 
        csv_val = './selected_topics_project/data/csv_files/valid.csv'
        csv_test = './selected_topics_project/data/csv_files/test.csv' 
        transform =  SimCLRDataTransform(data_augment)
        ''' 
        self.csv_path = csv_path 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)
        self.is_train = is_train

    def get_data_loaders(self):
        if self.is_train is True: 
            data_augment = self._get_simclr_pipeline_transform()
            dataset = CheXpertDataSet(
                self.csv_path, 
                transform=SimCLRDataTransform(data_augment), 
                policy="ones")  # return (xi, xj), label 

            loader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers, 
                drop_last=True)  
            
            return loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(size=self.input_shape[0]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.06 * self.input_shape[0])),
                transforms.ToTensor()
                ])
        return data_transforms



class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj




if __name__=="__main__":
    csv_path = './selected_topics_project/data/csv_files/train.csv'

    train_dataloader = SimCLRDataloader(
        batch_size=32, 
        num_workers=1, 
        input_shape="(224, 224, 3)", 
        s=1, 
        csv_path=csv_path).get_data_loaders()  
            batch_sample = next(iter(train_dataloader))
        inputs, labels = batch_sample
        print(inputs[0].shape, inputs[1].shape, labels.shape)  