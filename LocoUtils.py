import os
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import CocoDetection
from PIL import Image
from pycocotools.coco import COCO


'''
3=>0 small_load_carrier
5=>1 forklift
7=>2 pallet
10=>3 stillage
11=>4 pallet_truck
'''

class LocoDetection(torch.utils.data.Dataset):
    def __init__(self, root: str, annFile:str, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels
        label_mapping = {3: 0, 5: 1, 7: 2, 10: 3, 11: 4}
        labels = []
        for i in range(num_objs):
            labels.append(label_mapping[coco_annotation[i]['category_id']])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        #labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


class LocoDatasetLoader:
    def __init__(self, base_img_folder, annotations_folder):
        self.base_img_folder = base_img_folder
        self.annotations_folder = annotations_folder
        self.annotation_files = {
            'all': 'loco-all-v1.json',
            'subset-1': 'loco-sub1-v1-val.json',
            'subset-2': 'loco-sub2-v1-train.json',
            'subset-3': 'loco-sub3-v1-train.json',
            'subset-4': 'loco-sub4-v1-val.json',
            'subset-5': 'loco-sub5-v1-train.json'
        }
        self.datasets = {}
        self.load_datasets()

    def get_transform(self):
        custom_transforms = []
        custom_transforms.append(transforms.ToTensor())
        return transforms.Compose(custom_transforms)

    def load_datasets(self):
        # Load the complete set
        self.datasets['all'] = LocoDetection(
            root=self.base_img_folder,
            annFile=f'{self.annotations_folder}/{self.annotation_files["all"]}',
            transforms=self.get_transform()
        )

        # Define the subsets
        subsets = ['subset-1', 'subset-2', 'subset-3', 'subset-4', 'subset-5']
        
        # Load all subsets
        for subset in subsets:
            self.datasets[subset] = LocoDetection(
                root=self.base_img_folder,
                annFile=f'{self.annotations_folder}/{self.annotation_files[subset]}',
                transforms=self.get_transform()
            )

    def get_dataset(self, key):
        return self.datasets.get(key, None)
    
    # collate_fn needs for batch
    def collate_fn(self, batch):
            return tuple(zip(*batch))
    
    def get_train(self, batch_size):
        # select device (whether GPU or CPU)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        combined_train_dataset = ConcatDataset([self.get_dataset('subset-2'), self.get_dataset('subset-3'), self.get_dataset('subset-5')])

        # Create data loaders
        # Assuming datasets is the dictionary created earlier
        data_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=self.collate_fn)
        return data_loader
    
    def get_val(self, batch_size):     
        val_data_loader = DataLoader(self.get_dataset('subset-1'), batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=self.collate_fn)
        return val_data_loader
    
    def get_test(self, batch_size):
        test_data_loader = DataLoader(self.get_dataset('subset-4'), batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=self.collate_fn)
        return test_data_loader