from torch.utils.data import Dataset
import csv
from PIL import Image
import torch
class CheXpertDataSet(Dataset):
    def __init__(self, data_folder, csv_path, transform = None, policy = "ones"):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels.
        """
        image_names = []
        labels = []

        with open(csv_path, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None) # skip the header
            
            
            row_counter = 0  # Initialize a counter for rows
            for line in csvReader:
                if row_counter >= 300:  # Stop after 100 rows
                    break
    
            
            
            for line in csvReader:
                image_name = line[0]
                label = line[5:]

                for i in range(14):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == "ones":
                                label[i] = 1
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0

                image_names.append(data_folder + image_name)
                labels.append(label)
                
                row_counter += 1 

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)