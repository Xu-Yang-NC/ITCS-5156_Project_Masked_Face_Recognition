
import sys
import os

sys.path.append(os.getcwd())

from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import torch
import dlib
import cv2

from config_notmask import config


# training data
class TrainDataset(Dataset):
    def __init__(self, face_dir, mask_dir, csv_name, num_triplets, predicter_path, img_size,
                 training_triplets_path=None, transform=None):
        # Initialization
        self.df = pd.read_csv(csv_name, dtype={'id': object, 'name': object, 'class': int})
        self.face_dir = face_dir
        self.mask_dir = mask_dir
        self.num_triplets = num_triplets
        self.transform = transform

        # self.detector = dlib.get_frontal_face_detector()
        # self.predictor = dlib.shape_predictor(predicter_path)
        # self.img_size = img_size

        # Generate triplets data if there is none
        if os.path.exists(training_triplets_path):
            print("\nload {} triplets...".format(num_triplets))
            self.training_triplets = np.load(training_triplets_path)
            print('{} triplets loaded!'.format(num_triplets))
        else:
            self.training_triplets = self.generate_triplets(self, self.df, self.num_triplets)

    # Static method
    @staticmethod
    def generate_triplets(self, df, num_triplets):
        '''
        Generate triplets data
	Each triplets data include anchor sample, positive sample, negative sample

        Input： List info of original data and the required pair number
        '''
        print("\nGenerating {} triplets:".format(num_triplets))

        def make_dictionary_for_face_class(df):
            # df includes：id,name,class
            face_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in face_classes:
                    face_classes[label] = []
                #  Add the class to the dictionary
                face_classes[label].append(df.iloc[idx, 0])
            # face_classes = {'class0': [class0_id0, class0_id1, ...], 'class1': [class1_id0, ...], ...}
            return face_classes

        triplets = []
        classes = df['class'].unique()
        print("Generating face_classes...")
        face_classes = make_dictionary_for_face_class(df)
        print("Generating npy file...")
        # Program bar
        # progress_bar = tqdm(range(num_triplets))
        # for _ in progress_bar:
        progress_bar = tqdm(range(num_triplets))
        for _ in progress_bar:

            # Random pick two classes as positive and negative class
            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            # Raddom pick a picture form the positive class, if less then 2, pick again.
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            # If the positive class is equal to the negative class, pick another negative class.
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

           
            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
    
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
                
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))

            ineg = np.random.randint(0, len(face_classes[neg_class]))

            triplets.append(
                [
                    face_classes[pos_class][ianc],  
                    face_classes[pos_class][ipos],  
                    face_classes[neg_class][ineg],  
                    pos_class,  
                    neg_class,  
                    pos_name,  
                    neg_name  
                ]
            )

        print("Saving training triplets list in datasets/ directory ...")
        np.save(config['train_triplets_path'], triplets)
        print("Training triplets' list Saved!\n")

        return triplets

    

    def __len__(self):
        return len(self.training_triplets)

    def add_extension(self, path):
    
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        else:
            raise RuntimeError('No file "%s" with extension png or jpg.' % path)

    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

        anc_img = self.add_extension(os.path.join(self.face_dir, str(pos_name), str(anc_id)))
        pos_img = self.add_extension(os.path.join(self.face_dir, str(pos_name), str(pos_id)))
        neg_img = self.add_extension(os.path.join(self.face_dir, str(neg_name), str(neg_id)))

        anc_mask = self.add_extension(os.path.join(self.mask_dir, str(pos_name), str(anc_id)))
        pos_mask = self.add_extension(os.path.join(self.mask_dir, str(pos_name), str(pos_id)))
        neg_mask = self.add_extension(os.path.join(self.mask_dir, str(neg_name), str(neg_id)))
       
        anc_img = Image.open(anc_img).convert('RGB')
        pos_img = Image.open(pos_img).convert('RGB')
        neg_img = Image.open(neg_img).convert('RGB')

        anc_mask = cv2.imread(anc_mask, cv2.IMREAD_GRAYSCALE)
        pos_mask = cv2.imread(pos_mask, cv2.IMREAD_GRAYSCALE)
        neg_mask = cv2.imread(neg_mask, cv2.IMREAD_GRAYSCALE)

       
        # transform class to tensor
        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {
            'anc_img': anc_img,
            'pos_img': pos_img,
            'neg_img': neg_img,

            'mask_anc': anc_mask,
            'mask_pos': pos_mask,
            'mask_neg': neg_mask,

            'pos_class': pos_class,
            'neg_class': neg_class
        }

    
        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample












