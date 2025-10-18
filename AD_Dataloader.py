import os
import pandas as pd
import PIL.Image as Image
from torch.utils.data import Dataset
import torch
import h5py


class AD_Dataloader(Dataset):
    def __init__(self, root, data_list, label, transform=None):  
        self.root = root
        self.data = data_list 
        self.label = label
        self.transform = transform
        self.taskmaps = self.load_taskmaps(os.path.join(root, 'taskmaps'))
        self.gazeheatmap = ["presaccade1_l.png", "presaccade1_r.png", "presaccade2_l.png", "presaccade2_r.png", 
                            "antisaccade1_l.png", "antisaccade1_r.png", "antisaccade2_l.png", "antisaccade2_r.png", 
                            "sensitivity1_l.png", "sensitivity1_r.png", "sensitivity2_l.png", "sensitivity2_r.png", 
                            "sensitivity3_l.png", "sensitivity3_r.png", "saliency1_l.png", "saliency1_r.png", 
                            "saliency2_l.png", "saliency2_r.png", "saliency3_l.png", "saliency3_r.png", 
                            "saliency4_l.png", "saliency4_r.png", "saliency5_l.png", "saliency5_r.png"]
        
    def load_taskmaps(self, taskmap_path):
        taskmap_images = []
        for i in range(1,13):  # 假设有 16 张 taskmap 图片
            img_path = os.path.join(taskmap_path, f'{i}.png')  # 根据实际情况修改文件名格式
            with Image.open(img_path) as img:
                if self.transform is not None:
                    img = self.transform(img)
                taskmap_images.append(img)
        return taskmap_images
    
    def __getitem__(self, idx):
        mmse = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)
        moca = torch.tensor(self.data.iloc[idx, 2], dtype=torch.float32)
        other = torch.tensor(self.data.iloc[idx, 3:8], dtype=torch.float32)
        age_edu = torch.tensor(self.data.iloc[idx, 8:10], dtype=torch.float32)
        
        # disease = torch.tensor(self.data.iloc[idx, 10], dtype=torch.float32)

        if self.label == 'mmse':
            target = mmse
        elif self.label == 'moca':
            target = moca
        elif self.label == 'demo':
            target = mmse
        elif self.label == 'cls':
            icls = torch.tensor(self.data.iloc[idx, 11], dtype=torch.float32)
            target = icls
        else:
            raise ValueError('label must be mmse, moca, or fuse')
        # age_edu = torch.tensor(self.data.iloc[idx, 3:5], dtype=torch.float32) 
        # print(self.data.iloc[idx, 0])
        gazeheat_path = os.path.join(self.root, self.data.iloc[idx, 0], 'gazeheat')
        
        imgs = []
        for gh in self.gazeheatmap:
            img_path = os.path.join(gazeheat_path, gh)
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        # print(gazeheat_path)
        return imgs, self.taskmaps, age_edu, target

    def __len__(self):
        return len(self.data)


class AD2_Dataloader(Dataset):
    def __init__(self, root, data_list, label, transform=None, hdf5_file=None):  
        self.root = root
        self.data = data_list 
        self.label = label
        self.transform = transform
        self.hdf5_file = hdf5_file or os.path.join(root, 'ad.hdf5')
        self.gazeheatmap = ["presaccade1_l.png", "presaccade1_r.png", "presaccade2_l.png", "presaccade2_r.png", 
                            "antisaccade1_l.png", "antisaccade1_r.png", "antisaccade2_l.png", "antisaccade2_r.png", 
                            "sensitivity1_l.png", "sensitivity1_r.png", "sensitivity2_l.png", "sensitivity2_r.png", 
                            "sensitivity3_l.png", "sensitivity3_r.png", "saliency1_l.png", "saliency1_r.png", 
                            "saliency2_l.png", "saliency2_r.png", "saliency3_l.png", "saliency3_r.png", 
                            "saliency4_l.png", "saliency4_r.png", "saliency5_l.png", "saliency5_r.png"]
        if not os.path.exists(self.hdf5_file):
            self.create_hdf5(self.hdf5_file, self.data)
        self.hdf5_data = h5py.File(self.hdf5_file, 'r')
        self.taskmaps = [torch.tensor(self.hdf5_data['taskmaps'][str(i)][()], dtype=torch.float32) for i in range(1, 13)]
    
    def create_hdf5(self, file_path, data):
        with h5py.File(file_path, 'w') as file:
            # Create datasets for images and labels
            gaze_images = file.create_group('gaze_images')
            taskmaps = file.create_group('taskmaps')

            # Save taskmaps
            for i in range(1, 13):
                img_path = os.path.join(self.root, 'taskmaps', f'{i}.png')
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)
                taskmaps.create_dataset(str(i), data=img.numpy())

            # Iterate over rows in the DataFrame to save gaze heatmap images
            for index, row in data.iterrows():
                gazeheat_path = os.path.join(self.root, row.iloc[0], 'gazeheat')
                for idx, gh in enumerate(self.gazeheatmap):
                    img_path = os.path.join(gazeheat_path, gh)
                    img = Image.open(img_path)
                    if self.transform:
                        img = self.transform(img)
                    gaze_images.create_dataset(f'{index}_{idx}', data=img.numpy())

            # Save other data
            file.create_dataset('image_paths', data=data.iloc[:, 0].astype('S'))  # 存储图像路径
            file.create_dataset('mmse', data=data.iloc[:, 1].to_numpy(), dtype='f')
            file.create_dataset('moca', data=data.iloc[:, 2].to_numpy(), dtype='f')
            file.create_dataset('other', data=data.iloc[:, 3:8].to_numpy(), dtype='f')
            file.create_dataset('age_edu', data=data.iloc[:, 8:10].to_numpy(), dtype='f')
            file.create_dataset('icls', data=data.iloc[:, 11].to_numpy(), dtype='f')
    
    def __getitem__(self, idx):
        mmse = torch.tensor(self.hdf5_data['mmse'][idx], dtype=torch.float32)
        moca = torch.tensor(self.hdf5_data['moca'][idx], dtype=torch.float32)
        other = torch.tensor(self.hdf5_data['other'][idx], dtype=torch.float32)
        age_edu = torch.tensor(self.hdf5_data['age_edu'][idx], dtype=torch.float32)
        icls = torch.tensor(self.hdf5_data['icls'][idx], dtype=torch.float32)

        if self.label == 'mmse':
            target = mmse
        elif self.label == 'moca':
            target = moca
        elif self.label == 'demo':
            target = mmse  # Demo 也使用 MMSE 作为目标
        elif self.label == 'cls':
            target = icls
        else:
            raise ValueError('Invalid label specified')
        imgs = [torch.tensor(self.hdf5_data['gaze_images'][f'{idx}_{i}'][()], dtype=torch.float32) for i in range(len(self.gazeheatmap))]
        image_path = self.hdf5_data['image_paths'][idx].astype('str')
        # print(gazeheat_path)
        return imgs, self.taskmaps, age_edu, target, image_path

    def __len__(self):
        return len(self.data)


class ADC_Dataloader(Dataset):
    def __init__(self, root, data_list, label, transform=None, hdf5_file=None):  
        self.root = root
        self.data = data_list 
        self.label = label
        self.transform = transform
        self.hdf5_file = os.path.join(root, hdf5_file)
        self.gazeheatmap = ["presaccade1_l.png", "presaccade1_r.png", "presaccade2_l.png", "presaccade2_r.png", 
                            "antisaccade1_l.png", "antisaccade1_r.png", "antisaccade2_l.png", "antisaccade2_r.png", 
                            "sensitivity1_l.png", "sensitivity1_r.png", "sensitivity2_l.png", "sensitivity2_r.png", 
                            "sensitivity3_l.png", "sensitivity3_r.png", "saliency1_l.png", "saliency1_r.png", 
                            "saliency2_l.png", "saliency2_r.png", "saliency3_l.png", "saliency3_r.png", 
                            "saliency4_l.png", "saliency4_r.png", "saliency5_l.png", "saliency5_r.png", 
                            "color1_l.png", "color1_r.png", "color2_l.png", "color2_r.png", 
                            "color3_l.png", "color3_r.png", "color4_l.png", "color4_r.png"]
        if not os.path.exists(self.hdf5_file):
            self.create_hdf5(self.hdf5_file, self.data)
        self.hdf5_data = h5py.File(self.hdf5_file, 'r')
        self.taskmaps = [torch.tensor(self.hdf5_data['taskmaps'][str(i)][()], dtype=torch.float32) for i in range(1, 17)]
    
    def create_hdf5(self, file_path, data):
        with h5py.File(file_path, 'w') as file:
            # Create datasets for images and labels
            gaze_images = file.create_group('gaze_images')
            taskmaps = file.create_group('taskmaps')

            # Save taskmaps
            for i in range(1, 17):
                img_path = os.path.join(self.root, 'taskmaps', f'{i}.png')
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)
                taskmaps.create_dataset(str(i), data=img.numpy())

            # Iterate over rows in the DataFrame to save gaze heatmap images
            for index, row in data.iterrows():
                gazeheat_path = os.path.join(self.root, row.iloc[0], 'gazeheat')
                for idx, gh in enumerate(self.gazeheatmap):
                    img_path = os.path.join(gazeheat_path, gh)
                    img = Image.open(img_path)
                    if self.transform:
                        img = self.transform(img)
                    gaze_images.create_dataset(f'{index}_{idx}', data=img.numpy())

            # Save other data
            file.create_dataset('image_paths', data=data.iloc[:, 0].astype('S'))  # 存储图像路径
            file.create_dataset('mmse', data=data.iloc[:, 1].to_numpy(), dtype='f')
            file.create_dataset('moca', data=data.iloc[:, 2].to_numpy(), dtype='f')
            file.create_dataset('other', data=data.iloc[:, 3:8].to_numpy(), dtype='f')
            file.create_dataset('age_edu', data=data.iloc[:, 8:10].to_numpy(), dtype='f')
            file.create_dataset('icls', data=data.iloc[:, 11].to_numpy(), dtype='f')
    
    def __getitem__(self, idx):
        mmse = torch.tensor(self.hdf5_data['mmse'][idx], dtype=torch.float32)
        moca = torch.tensor(self.hdf5_data['moca'][idx], dtype=torch.float32)
        other = torch.tensor(self.hdf5_data['other'][idx], dtype=torch.float32)
        age_edu = torch.tensor(self.hdf5_data['age_edu'][idx], dtype=torch.float32)
        icls = torch.tensor(self.hdf5_data['icls'][idx], dtype=torch.float32)

        if self.label == 'mmse':
            target = mmse
        elif self.label == 'moca':
            target = moca
        elif self.label == 'demo':
            target = mmse  # Demo 也使用 MMSE 作为目标
        elif self.label == 'cls':
            target = icls
        else:
            raise ValueError('Invalid label specified')
        imgs = [torch.tensor(self.hdf5_data['gaze_images'][f'{idx}_{i}'][()], dtype=torch.float32) for i in range(len(self.gazeheatmap))]
        image_path = self.hdf5_data['image_paths'][idx].astype('str')
        # print(gazeheat_path)
        return imgs, self.taskmaps, age_edu, target, image_path

    def __len__(self):
        return len(self.data)
    

class ADNC_Dataloader(Dataset):
    def __init__(self, root, data_list, label, transform=None):  
        self.root = root
        self.data = data_list 
        self.label = label
        self.transform = transform
        self.taskmaps = self.load_taskmaps(os.path.join(root, 'taskmaps'))
        self.gazeheatmap = ["presaccade1_l.png", "presaccade1_r.png", "presaccade2_l.png", "presaccade2_r.png", 
                            "antisaccade1_l.png", "antisaccade1_r.png", "antisaccade2_l.png", "antisaccade2_r.png", 
                            "sensitivity1_l.png", "sensitivity1_r.png", "sensitivity2_l.png", "sensitivity2_r.png", 
                            "sensitivity3_l.png", "sensitivity3_r.png", "saliency1_l.png", "saliency1_r.png", 
                            "saliency2_l.png", "saliency2_r.png", "saliency3_l.png", "saliency3_r.png", 
                            "saliency4_l.png", "saliency4_r.png", "saliency5_l.png", "saliency5_r.png"]
        
    def load_taskmaps(self, taskmap_path):
        taskmap_images = []
        for i in range(1,13):  # 假设有 16 张 taskmap 图片
            img_path = os.path.join(taskmap_path, f'{i}.png')  # 根据实际情况修改文件名格式
            with Image.open(img_path) as img:
                if self.transform is not None:
                    img = self.transform(img)
                taskmap_images.append(img)
        return taskmap_images
    
    def __getitem__(self, idx):
        mmse = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)
        moca = torch.tensor(self.data.iloc[idx, 2], dtype=torch.float32)
        other = torch.tensor(self.data.iloc[idx, 3:8], dtype=torch.float32)
        age_edu = torch.tensor(self.data.iloc[idx, 8:10], dtype=torch.float32)
        
        # disease = torch.tensor(self.data.iloc[idx, 10], dtype=torch.float32)

        if self.label == 'mmse':
            target = mmse
            weight = 2.0 if mmse < 24 else 1.0
        elif self.label == 'moca':
            target = moca
            weight = 2.0 if moca < 24 else 1.0
        elif self.label == 'demo':
            target = mmse
        elif self.label == 'cls':
            icls = torch.tensor(self.data.iloc[idx, 11], dtype=torch.float32)
            target = icls
        else:
            raise ValueError('label must be mmse, moca, or fuse')
        # age_edu = torch.tensor(self.data.iloc[idx, 3:5], dtype=torch.float32) 
        # print(self.data.iloc[idx, 0])
        gazeheat_path = os.path.join(self.root, self.data.iloc[idx, 0], 'gazeheat')
        
        imgs = []
        for gh in self.gazeheatmap:
            img_path = os.path.join(gazeheat_path, gh)
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        # print(gazeheat_path)
        return imgs, self.taskmaps, age_edu, target, weight

    def __len__(self):
        return len(self.data)


class ADNC2_Dataloader(Dataset):
    def __init__(self, root, data_list, label, transform=None, hdf5_file=None):  
        self.root = root
        self.data = data_list 
        self.label = label
        self.transform = transform
        self.hdf5_file = hdf5_file or os.path.join(root, 'data.hdf5')
        self.gazeheatmap = ["presaccade1_l.png", "presaccade1_r.png", "presaccade2_l.png", "presaccade2_r.png", 
                            "antisaccade1_l.png", "antisaccade1_r.png", "antisaccade2_l.png", "antisaccade2_r.png", 
                            "sensitivity1_l.png", "sensitivity1_r.png", "sensitivity2_l.png", "sensitivity2_r.png", 
                            "sensitivity3_l.png", "sensitivity3_r.png", "saliency1_l.png", "saliency1_r.png", 
                            "saliency2_l.png", "saliency2_r.png", "saliency3_l.png", "saliency3_r.png", 
                            "saliency4_l.png", "saliency4_r.png", "saliency5_l.png", "saliency5_r.png"]
        if not os.path.exists(self.hdf5_file):
            self.create_hdf5(self.hdf5_file, self.data)
        self.hdf5_data = h5py.File(self.hdf5_file, 'r')
        self.taskmaps = [torch.tensor(self.hdf5_data['taskmaps'][str(i)][()], dtype=torch.float32) for i in range(1, 13)]
    
    def create_hdf5(self, file_path, data):
        with h5py.File(file_path, 'w') as file:
            # Create datasets for images and labels
            gaze_images = file.create_group('gaze_images')
            taskmaps = file.create_group('taskmaps')

            # Save taskmaps
            for i in range(1, 13):
                img_path = os.path.join(self.root, 'taskmaps', f'{i}.png')
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)
                taskmaps.create_dataset(str(i), data=img.numpy())

            # Iterate over rows in the DataFrame to save gaze heatmap images
            for index, row in data.iterrows():
                gazeheat_path = os.path.join(self.root, row.iloc[0], 'gazeheat')
                for idx, gh in enumerate(self.gazeheatmap):
                    img_path = os.path.join(gazeheat_path, gh)
                    img = Image.open(img_path)
                    if self.transform:
                        img = self.transform(img)
                    gaze_images.create_dataset(f'{index}_{idx}', data=img.numpy())

            # Save other data
            file.create_dataset('image_paths', data=data.iloc[:, 0].astype('S'))  # 存储图像路径
            file.create_dataset('mmse', data=data.iloc[:, 1].to_numpy(), dtype='f')
            file.create_dataset('moca', data=data.iloc[:, 2].to_numpy(), dtype='f')
            file.create_dataset('other', data=data.iloc[:, 3:8].to_numpy(), dtype='f')
            file.create_dataset('age_edu', data=data.iloc[:, 8:10].to_numpy(), dtype='f')
            file.create_dataset('icls', data=data.iloc[:, 11].to_numpy(), dtype='f')
    
    def __getitem__(self, idx):
        mmse = torch.tensor(self.hdf5_data['mmse'][idx], dtype=torch.float32)
        moca = torch.tensor(self.hdf5_data['moca'][idx], dtype=torch.float32)
        other = torch.tensor(self.hdf5_data['other'][idx], dtype=torch.float32)
        age_edu = torch.tensor(self.hdf5_data['age_edu'][idx], dtype=torch.float32)
        icls = torch.tensor(self.hdf5_data['icls'][idx], dtype=torch.float32)

        if self.label == 'mmse':
            target = mmse
        elif self.label == 'moca':
            target = moca
        elif self.label == 'demo':
            target = mmse  # Demo 也使用 MMSE 作为目标
        elif self.label == 'cls':
            target = icls
        else:
            raise ValueError('Invalid label specified')
        weight = 2.0 if target < 24 else 1.0
        imgs = [torch.tensor(self.hdf5_data['gaze_images'][f'{idx}_{i}'][()], dtype=torch.float32) for i in range(len(self.gazeheatmap))]
        image_path = self.hdf5_data['image_paths'][idx].astype('str')
        # print(gazeheat_path)
        return imgs, self.taskmaps, age_edu, target, weight, image_path

    def __len__(self):
        return len(self.data)


class ADF_Dataloader(Dataset):
    def __init__(self, root, data_list, label, transform=None):  
        self.root = root
        self.data = data_list 
        self.label = label
        self.transform = transform
        self.taskmaps = self.load_taskmaps(os.path.join(root, 'taskmaps'))
        self.gazeheatmap = ["presaccade1_l.png", "presaccade1_r.png", "presaccade2_l.png", "presaccade2_r.png", 
                            "antisaccade1_l.png", "antisaccade1_r.png", "antisaccade2_l.png", "antisaccade2_r.png", 
                            "sensitivity1_l.png", "sensitivity1_r.png", "sensitivity2_l.png", "sensitivity2_r.png", 
                            "sensitivity3_l.png", "sensitivity3_r.png", "saliency1_l.png", "saliency1_r.png", 
                            "saliency2_l.png", "saliency2_r.png", "saliency3_l.png", "saliency3_r.png", 
                            "saliency4_l.png", "saliency4_r.png", "saliency5_l.png", "saliency5_r.png"]
        
    def load_taskmaps(self, taskmap_path):
        taskmap_images = []
        for i in range(1,13):  # 假设有 16 张 taskmap 图片
            img_path = os.path.join(taskmap_path, f'{i}.png')  # 根据实际情况修改文件名格式
            with Image.open(img_path) as img:
                if self.transform is not None:
                    img = self.transform(img)
                taskmap_images.append(img)
        return taskmap_images

    def __getitem__(self, idx):
        mmse = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)
        moca = torch.tensor(self.data.iloc[idx, 2], dtype=torch.float32)
        other = torch.tensor(self.data.iloc[idx, 3:8], dtype=torch.float32)
        age_edu = torch.tensor(self.data.iloc[idx, 8:10], dtype=torch.float32)
        icls = torch.tensor(self.data.iloc[idx, 11], dtype=torch.float32)
        # disease = torch.tensor(self.data.iloc[idx, 10], dtype=torch.float32)

        if self.label == 'mmse':
            target = mmse
        elif self.label == 'moca':
            target = moca
        else:
            raise ValueError('label must be mmse, moca, or fuse')
        # age_edu = torch.tensor(self.data.iloc[idx, 3:5], dtype=torch.float32) 
        # print(self.data.iloc[idx, 0])
        gazeheat_path = os.path.join(self.root, self.data.iloc[idx, 0], 'gazeheat')
        
        imgs = []
        for gh in self.gazeheatmap:
            img_path = os.path.join(gazeheat_path, gh)
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        # print(gazeheat_path)
        return imgs, self.taskmaps, age_edu, icls, target 

    def __len__(self):
        return len(self.data)
          

class Test_dataloader(Dataset):
    def __init__(self, root, data_list, label, transform=None, hdf5_file=None):  
        self.root = root
        self.data = data_list 
        self.label = label
        self.transform = transform
        self.hdf5_file = os.path.join(root, hdf5_file)
        self.gazeheatmap = ["presaccade1_l.png", "presaccade1_r.png", "presaccade2_l.png", "presaccade2_r.png", 
                            "antisaccade1_l.png", "antisaccade1_r.png", "antisaccade2_l.png", "antisaccade2_r.png", 
                            "sensitivity1_l.png", "sensitivity1_r.png", "sensitivity2_l.png", "sensitivity2_r.png", 
                            "sensitivity3_l.png", "sensitivity3_r.png", "saliency1_l.png", "saliency1_r.png", 
                            "saliency2_l.png", "saliency2_r.png", "saliency3_l.png", "saliency3_r.png", 
                            "saliency4_l.png", "saliency4_r.png", "saliency5_l.png", "saliency5_r.png"]
        if not os.path.exists(self.hdf5_file):
            self.create_hdf5(self.hdf5_file, self.data)
        self.hdf5_data = h5py.File(self.hdf5_file, 'r')
        self.taskmaps = [torch.tensor(self.hdf5_data['taskmaps'][str(i)][()], dtype=torch.float32) for i in range(1, 13)]
    
    def create_hdf5(self, file_path, data):
        with h5py.File(file_path, 'w') as file:
            # Create datasets for images and labels
            gaze_images = file.create_group('gaze_images')
            taskmaps = file.create_group('taskmaps')

            # Save taskmaps
            for i in range(1, 13):
                img_path = os.path.join(self.root, 'taskmaps', f'{i}.png')
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)
                taskmaps.create_dataset(str(i), data=img.numpy())

            # Iterate over rows in the DataFrame to save gaze heatmap images
            for index, row in data.iterrows():
                gazeheat_path = os.path.join(self.root, row.iloc[0], 'gazeheat')
                for idx, gh in enumerate(self.gazeheatmap):
                    img_path = os.path.join(gazeheat_path, gh)
                    img = Image.open(img_path)
                    if self.transform:
                        img = self.transform(img)
                    gaze_images.create_dataset(f'{index}_{idx}', data=img.numpy())

            # Save other data
            file.create_dataset('image_paths', data=data.iloc[:, 0].astype('S'))  # 存储图像路径
            file.create_dataset('mmse', data=data.iloc[:, 1].to_numpy(), dtype='f')
            file.create_dataset('moca', data=data.iloc[:, 2].to_numpy(), dtype='f')
            file.create_dataset('other', data=data.iloc[:, 3:8].to_numpy(), dtype='f')
            file.create_dataset('age_edu', data=data.iloc[:, 8:10].to_numpy(), dtype='f')
            file.create_dataset('icls', data=data.iloc[:, 11].to_numpy(), dtype='f')
    
    def __getitem__(self, idx):
        mmse = torch.tensor(self.hdf5_data['mmse'][idx], dtype=torch.float32)
        moca = torch.tensor(self.hdf5_data['moca'][idx], dtype=torch.float32)
        other = torch.tensor(self.hdf5_data['other'][idx], dtype=torch.float32)
        age_edu = torch.tensor(self.hdf5_data['age_edu'][idx], dtype=torch.float32)
        icls = torch.tensor(self.hdf5_data['icls'][idx], dtype=torch.float32)

        if self.label == 'mmse':
            target = mmse
        elif self.label == 'moca':
            target = moca
        elif self.label == 'cls':
            target = icls
        else:
            raise ValueError('Invalid label specified')
        imgs = [torch.tensor(self.hdf5_data['gaze_images'][f'{idx}_{i}'][()], dtype=torch.float32) for i in range(len(self.gazeheatmap))]
        image_path = self.hdf5_data['image_paths'][idx].astype('str')
        # print(gazeheat_path)
        return imgs, self.taskmaps, age_edu, target, icls

    def __len__(self):
        return len(self.data)
    

import os
import pandas as pd
from PIL import Image
import numpy as np

class ML_Dataloader():
    def __init__(self, root, data_list, label, transform=None):  
        self.root = root
        self.data = data_list 
        self.label = label
        self.transform = transform

    def load_data(self):
        features = []
        targets = []

        for idx in range(len(self.data)):
            mmse = self.data.iloc[idx, 1]
            moca = self.data.iloc[idx, 2]
            other = self.data.iloc[idx, 3:8].values
            age_edu = self.data.iloc[idx, 8:10].values

            if self.label == 'mmse':
                target = mmse
            elif self.label == 'moca':
                target = moca
            elif self.label == 'cls':
                icls = self.data.iloc[idx, 12]
                target = icls
            else:
                raise ValueError('Label must be mmse, moca, or cls')
            
            feature_vector = np.concatenate([other, age_edu])
            features.append(feature_vector)
            targets.append(target)

        return np.array(features), np.array(targets)
    
import networkx as nx
from torch_geometric.data import Data, Batch
import pickle

class ADG_Dataloader(Dataset):
    def __init__(self, root, data_list, label, transform=None, save_file='graph_data.pt'):
        self.root = root
        self.data = data_list 
        self.label = label
        self.transform = transform
        self.save_file = os.path.join(self.root, save_file)
        self.gazegraphmap = ["presaccade1_l.gpickle", "presaccade1_r.gpickle", "presaccade2_l.gpickle", "presaccade2_r.gpickle", 
                            "antisaccade1_l.gpickle", "antisaccade1_r.gpickle", "antisaccade2_l.gpickle", "antisaccade2_r.gpickle", 
                            "sensitivity1_l.gpickle", "sensitivity1_r.gpickle", "sensitivity2_l.gpickle", "sensitivity2_r.gpickle", 
                            "sensitivity3_l.gpickle", "sensitivity3_r.gpickle", "saliency1_l.gpickle", "saliency1_r.gpickle", 
                            "saliency2_l.gpickle", "saliency2_r.gpickle", "saliency3_l.gpickle", "saliency3_r.gpickle", 
                            "saliency4_l.gpickle", "saliency4_r.gpickle", "saliency5_l.gpickle", "saliency5_r.gpickle"]
        if not os.path.exists(self.save_file):
            self.create_data_file(self.save_file, self.data)
        self.data_dict = torch.load(self.save_file)
        self.taskmaps = [torch.tensor(self.data_dict['taskmaps'][i-1], dtype=torch.float32) for i in range(1, 13)]
    
    def create_data_file(self, file_path, data):
        data_dict = {'graphs': [], 'taskmaps': [], 'image_paths': [], 'mmse': [], 'moca': [], 'other': [], 'age_edu': [], 'icls': []}

        # 处理 taskmaps
        for i in range(1, 13):
            img_path = os.path.join(self.root, 'taskmaps', f'{i}.png')
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            data_dict['taskmaps'].append(img.numpy())

        # 处理图数据和其他特征数据
        for index, row in data.iterrows():
            gazegraph_path = os.path.join(self.root, row.iloc[0], 'gazegraph')
            graphs = []
            for gg in self.gazegraphmap:
                graph_path = os.path.join(gazegraph_path, gg)
                with open(graph_path, 'rb') as f:
                    G = pickle.load(f)

                # 获取节点的位置信息
                positions = nx.get_node_attributes(G, 'pos')                
                # 获取节点的时间信息
                times = nx.get_node_attributes(G, 'frame_number')
                # 获取边信息
                edge_index = torch.tensor(list(G.edges)).t().contiguous()
                edge_weights = torch.tensor([G.edges[e]['weight'] for e in G.edges], dtype=torch.float32)
                edge_orientations = torch.tensor([G.edges[e]['orientation'] for e in G.edges], dtype=torch.float32)
                # 合并边特征
                edge_features = torch.stack((edge_weights, edge_orientations), dim=1)

                # 获取节点特征
                node_features = torch.tensor([positions[n] for n in G.nodes], dtype=torch.float32)
                
                # 将图数据存储为 PyG 的 Data 对象
                graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
                graph_data.edge_weights = edge_weights
                graph_data.edge_orientations = edge_orientations
                
                graphs.append(graph_data)

            # 存储所有图和其他信息
            data_dict['graphs'].append(graphs)
            data_dict['image_paths'].append(row.iloc[0])
            data_dict['mmse'].append(row.iloc[1])
            data_dict['moca'].append(row.iloc[2])
            data_dict['other'].append(row.iloc[3:8].tolist())
            data_dict['age_edu'].append(row.iloc[8:10].tolist())
            data_dict['icls'].append(row.iloc[11])

        torch.save(data_dict, file_path)
    
    def __getitem__(self, idx):
        graphs = self.data_dict['graphs'][idx]
        # 获取其他信息
        mmse = torch.tensor(self.data_dict['mmse'][idx], dtype=torch.float32)
        moca = torch.tensor(self.data_dict['moca'][idx], dtype=torch.float32)
        other = torch.tensor(self.data_dict['other'][idx], dtype=torch.float32)
        age_edu = torch.tensor(self.data_dict['age_edu'][idx], dtype=torch.float32)
        icls = torch.tensor(self.data_dict['icls'][idx], dtype=torch.float32)

        if self.label == 'mmse':
            target = mmse
        elif self.label == 'moca':
            target = moca
        elif self.label == 'demo':
            target = mmse
        elif self.label == 'cls':
            target = icls
        else:
            raise ValueError('Invalid label specified')
        
        image_path = self.data_dict['image_paths'][idx]
        return graphs, age_edu, target, image_path
    
    def custom_collate_fn(batch):
        batch_graphs = []  # 用于存储批处理后的图列表
        age_edu_list = []
        target_list = []
        image_path_list = []

        # batch 是一个列表，其中每个元素都是 (graphs, age_edu, target, image_path)
        # graphs 是一个列表，长度为 n
        num_graphs_per_sample = len(batch[0][0])  # 样本中图的数量 n

        # 初始化一个列表，其中每个元素将是一个 Batch 对象
        for i in range(num_graphs_per_sample):
            graphs_at_position_i = [sample[0][i] for sample in batch]  # 提取每个样本在位置 i 的图
            batch_graph = Batch.from_data_list(graphs_at_position_i)
            batch_graphs.append(batch_graph)

        # 收集其他信息
        for sample in batch:
            age_edu_list.append(sample[1])
            target_list.append(sample[2])
            image_path_list.append(sample[3])

        # 将 age_edu 和 target 转换为张量
        age_edu_batch = torch.stack(age_edu_list)
        target_batch = torch.stack(target_list)

        return batch_graphs, age_edu_batch, target_batch, image_path_list
    
    def __len__(self):
        return len(self.data)
    

class ADGH_Dataloader(Dataset):
    def __init__(self, root, data_list, label, transform=None, save_file='adgh.pt'):
        self.root = root
        self.data = data_list 
        self.label = label
        self.transform = transform
        self.save_file = os.path.join(self.root, save_file)
        self.gazegraphmap = ["presaccade1_l.pickle", "presaccade1_r.pickle", "presaccade2_l.pickle", "presaccade2_r.pickle", 
                            "antisaccade1_l.pickle", "antisaccade1_r.pickle", "antisaccade2_l.pickle", "antisaccade2_r.pickle", 
                            "sensitivity1_l.pickle", "sensitivity1_r.pickle", "sensitivity2_l.pickle", "sensitivity2_r.pickle", 
                            "sensitivity3_l.pickle", "sensitivity3_r.pickle", "saliency1_l.pickle", "saliency1_r.pickle", 
                            "saliency2_l.pickle", "saliency2_r.pickle", "saliency3_l.pickle", "saliency3_r.pickle", 
                            "saliency4_l.pickle", "saliency4_r.pickle", "saliency5_l.pickle", "saliency5_r.pickle"]
        self.gazeheatmap = ["presaccade1_l.png", "presaccade1_r.png", "presaccade2_l.png", "presaccade2_r.png", 
                            "antisaccade1_l.png", "antisaccade1_r.png", "antisaccade2_l.png", "antisaccade2_r.png", 
                            "sensitivity1_l.png", "sensitivity1_r.png", "sensitivity2_l.png", "sensitivity2_r.png", 
                            "sensitivity3_l.png", "sensitivity3_r.png", "saliency1_l.png", "saliency1_r.png", 
                            "saliency2_l.png", "saliency2_r.png", "saliency3_l.png", "saliency3_r.png", 
                            "saliency4_l.png", "saliency4_r.png", "saliency5_l.png", "saliency5_r.png"] 
        if not os.path.exists(self.save_file):
            self.create_data_file(self.save_file, self.data)
        self.data_dict = torch.load(self.save_file)
        self.taskmaps = [self.data_dict['taskmaps'][i-1].clone().detach().to(torch.float32) for i in range(1, 13)]  # {{ edit_1 }}    
    def create_data_file(self, file_path, data):
        data_dict = {'gazegraphs': [], 'gazeheats': [], 'taskmaps': [], 'image_paths': [], 'mmse': [], 'moca': [], 'other': [], 'age_edu': [], 'icls': []}

        # 处理 taskmaps
        for i in range(1, 13):
            img_path = os.path.join(self.root, 'taskmaps', f'{i}.png')
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            data_dict['taskmaps'].append(img)
            
        for index, row in data.iterrows():
            gazeheat_path = os.path.join(self.root, row.iloc[0], 'gazeheat')
            gazegraph_path = os.path.join(self.root, row.iloc[0], 'gazegraph')
            heatmaps = []            
            gazegraphs = []
            for gh in self.gazeheatmap:
                img_path = os.path.join(gazeheat_path, gh)
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)
                heatmaps.append(img)
            data_dict['gazeheats'].append(heatmaps)
            
            for gg in self.gazegraphmap:
                graph_path = os.path.join(gazegraph_path, gg)
                with open(graph_path, 'rb') as f:
                    graph_data = pickle.load(f)
                gazegraphs.append(graph_data)
            data_dict['gazegraphs'].append(gazegraphs)
        
            data_dict['image_paths'].append(row.iloc[0])
            data_dict['mmse'].append(row.iloc[1])
            data_dict['moca'].append(row.iloc[2])
            data_dict['other'].append(row.iloc[3:8].tolist())
            data_dict['age_edu'].append(row.iloc[8:10].tolist())
            data_dict['icls'].append(row.iloc[11])

        torch.save(data_dict, file_path)
    
    def __getitem__(self, idx):
        heatmaps = self.data_dict['gazeheats'][idx]
        graphs = self.data_dict['gazegraphs'][idx]
        # 获取其他信息
        mmse = torch.tensor(self.data_dict['mmse'][idx], dtype=torch.float32)
        moca = torch.tensor(self.data_dict['moca'][idx], dtype=torch.float32)
        other = torch.tensor(self.data_dict['other'][idx], dtype=torch.float32)
        age_edu = torch.tensor(self.data_dict['age_edu'][idx], dtype=torch.float32)
        icls = torch.tensor(self.data_dict['icls'][idx], dtype=torch.float32)

        if self.label == 'mmse':
            target = mmse
        elif self.label == 'moca':
            target = moca
        elif self.label == 'demo':
            target = mmse
        elif self.label == 'cls':
            target = icls
        else:
            raise ValueError('Invalid label specified')
        
        image_path = self.data_dict['image_paths'][idx]
        return graphs, heatmaps, self.taskmaps, age_edu, target, image_path
    
    def __len__(self):
        return len(self.data)
    
class ADGH_Dataloader2(Dataset):
    def __init__(self, root, data_list, label, transform=None, save_file='adgh.pt'):
        self.root = root
        self.data = data_list 
        self.label = label
        self.transform = transform
        self.save_file = os.path.join(self.root, save_file)
        self.gazegraphmap = ["presaccade1_l.pickle", "presaccade1_r.pickle", "presaccade2_l.pickle", "presaccade2_r.pickle", 
                            "antisaccade1_l.pickle", "antisaccade1_r.pickle", "antisaccade2_l.pickle", "antisaccade2_r.pickle", 
                            "sensitivity1_l.pickle", "sensitivity1_r.pickle", "sensitivity2_l.pickle", "sensitivity2_r.pickle", 
                            "sensitivity3_l.pickle", "sensitivity3_r.pickle", "saliency1_l.pickle", "saliency1_r.pickle", 
                            "saliency2_l.pickle", "saliency2_r.pickle", "saliency3_l.pickle", "saliency3_r.pickle", 
                            "saliency4_l.pickle", "saliency4_r.pickle", "saliency5_l.pickle", "saliency5_r.pickle"]
        self.gazeheatmap = ["presaccade1_l.png", "presaccade1_r.png", "presaccade2_l.png", "presaccade2_r.png", 
                            "antisaccade1_l.png", "antisaccade1_r.png", "antisaccade2_l.png", "antisaccade2_r.png", 
                            "sensitivity1_l.png", "sensitivity1_r.png", "sensitivity2_l.png", "sensitivity2_r.png", 
                            "sensitivity3_l.png", "sensitivity3_r.png", "saliency1_l.png", "saliency1_r.png", 
                            "saliency2_l.png", "saliency2_r.png", "saliency3_l.png", "saliency3_r.png", 
                            "saliency4_l.png", "saliency4_r.png", "saliency5_l.png", "saliency5_r.png"] 
        if not os.path.exists(self.save_file):
            self.create_data_file(self.save_file, self.data)
        self.data_dict = torch.load(self.save_file)
        self.taskmaps = [self.data_dict['taskmaps'][i-1].clone().detach().to(torch.float32) for i in range(1, 13)]  # {{ edit_1 }}    
    def create_data_file(self, file_path, data):
        data_dict = {'gazegraphs': [], 'gazeheats': [], 'taskmaps': [], 'image_paths': [], 'mmse': [], 'moca': [], 'other': [], 'age_edu': [], 'icls': []}

        # 处理 taskmaps
        for i in range(1, 13):
            img_path = os.path.join(self.root, 'taskmaps', f'{i}.png')
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            data_dict['taskmaps'].append(img)
            
        for index, row in data.iterrows():
            gazeheat_path = os.path.join(self.root, row.iloc[0], 'gazeheat')
            gazegraph_path = os.path.join(self.root, row.iloc[0], 'gazegraph')
            heatmaps = []            
            gazegraphs = []
            for gh in self.gazeheatmap:
                img_path = os.path.join(gazeheat_path, gh)
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)
                heatmaps.append(img)
            data_dict['gazeheats'].append(heatmaps)
            
            for gg in self.gazegraphmap:
                graph_path = os.path.join(gazegraph_path, gg)
                with open(graph_path, 'rb') as f:
                    graph_data = pickle.load(f)
                gazegraphs.append(graph_data)
            data_dict['gazegraphs'].append(gazegraphs)
        
            data_dict['image_paths'].append(row.iloc[0])
            data_dict['mmse'].append(row.iloc[1])
            data_dict['moca'].append(row.iloc[2])
            data_dict['other'].append(row.iloc[3:8].tolist())
            data_dict['age_edu'].append(row.iloc[8:10].tolist())
            data_dict['icls'].append(row.iloc[11])

        torch.save(data_dict, file_path)
    
    def __getitem__(self, idx):
        heatmaps = self.data_dict['gazeheats'][idx]
        graphs = self.data_dict['gazegraphs'][idx]
        # 获取其他信息
        mmse = torch.tensor(self.data_dict['mmse'][idx], dtype=torch.float32)
        moca = torch.tensor(self.data_dict['moca'][idx], dtype=torch.float32)
        other = torch.tensor(self.data_dict['other'][idx], dtype=torch.float32)
        age_edu = torch.tensor(self.data_dict['age_edu'][idx], dtype=torch.float32)
        icls = torch.tensor(self.data_dict['icls'][idx], dtype=torch.float32)

        if self.label == 'mmse':
            target = mmse
        elif self.label == 'moca':
            target = moca
        elif self.label == 'demo':
            target = mmse
        elif self.label == 'cls':
            target = icls
        else:
            raise ValueError('Invalid label specified')
        
        image_path = self.data_dict['image_paths'][idx]
        return graphs, heatmaps, self.taskmaps, age_edu, target, icls, image_path
    
    def __len__(self):
        return len(self.data)