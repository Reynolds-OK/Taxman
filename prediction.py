import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import albumentations as album
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.metrics



"""Helper Functions"""

# Center crop padded image / mask to original image dims
def crop_image(image, target_image_dims=[1500,1500,3]):

    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
        padding:image_size - padding,
        padding:image_size - padding,
        :,
    ]


# helper function for data visualization
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]);
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
#     print(semantic_map)
#     print(semantic_map.shape)
    return semantic_map

# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    x = np.argmax(image, axis = -1)
#     print(x)
#     print(x.shape)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

def get_training_augmentation():
    train_transform = [
        album.RandomCrop(height=256, width=256, always_apply=True),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0,value=0 ),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)

class BuildingsDataset(torch.utils.data.Dataset):

    """Massachusetts Buildings Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_dir,
            masks_dir,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
    ):

        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        # return length of
        return len(self.image_paths)
        
        
def count_files_in_directory(directory):
    """
    Count the number of files in the specified directory.

    Parameters:
    - directory: Path to the directory

    Returns:
    - The number of files in the directory
    """
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return len(files)



"""Images for Prediction"""
def run_predictions(output):

    """Load Model"""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('./model/unet++_model.pth',map_location=DEVICE)
    model.to(DEVICE)
    model.eval()
    
    DATA_DIR = 'capture/'

    # x_test_dir = os.path.join(DATA_DIR, 'test')
    # y_test_dir = os.path.join(DATA_DIR, 'test_labels')
    x_test_dir = os.path.join(DATA_DIR, 'starting')
    y_test_dir = os.path.join(DATA_DIR, 'starting')
    
    class_dict = pd.read_csv("model/label_class_dict.csv")
    # Get class names
    class_names = class_dict['name'].tolist()
    # Get class RGB values
    class_rgb_values = class_dict[['r','g','b']].values.tolist()
    
    print('All dataset classes and their corresponding RGB values in labels:')
    print('Class Names: ', class_names)
    print('Class RGB values: ', class_rgb_values)
    
    ENCODER = 'tu-tf_efficientnetv2_l'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = class_names
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    # Useful to shortlist specific classes in datasets with large number of classes
    select_classes = ['background', 'building']
    
    # Get RGB values of required classes
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]
    
    print('Selected classes and their corresponding RGB values in labels:')
    print('Class Names: ', class_names)
    print('Class RGB values: ', class_rgb_values)
    
    # create test dataloader (with preprocessing operation: to_tensor(...))
    test_dataset = BuildingsDataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )
    
    test_dataloader = DataLoader(test_dataset,batch_size=16)
    
    # test dataset for visualization (without preprocessing transformations)
    test_dataset_vis = BuildingsDataset(
        x_test_dir, y_test_dir,
        augmentation=get_validation_augmentation(),
        class_rgb_values=select_class_rgb_values,
    )
    
    sample_preds_folder = output
    if not os.path.exists(sample_preds_folder):
        os.makedirs(sample_preds_folder)
        print('done')                                                   
    


    # Get the number of files in the directory
    number_of_imgs= count_files_in_directory(x_test_dir)
    print(f"Number of files in '{x_test_dir}': {number_of_imgs}")
    
    # for idx in range(number_of_imgs):
    for idx in range(number_of_imgs):
        image, gt_mask = test_dataset[idx]
        # image_vis = crop_image(test_dataset_vis[idx][0].astype('uint8'))
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    
        # Predict test image
        pred_mask = model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
    
        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask,(1,2,0))
    
        # Get prediction channel corresponding to building
        # pred_road_heatmap = pred_mask[:,:,select_classes.index('building')]
        pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values))
    
        # Convert gt_mask from `CHW` format to `HWC` format
        # gt_mask = np.transpose(gt_mask,(1,2,0))
        # gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values))
        # cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"), np.hstack([image_vis, gt_mask, pred_mask])[:,:,::-1])
    
        # Save each image separately
        # cv2.imwrite(os.path.join(sample_preds_folder, f"image_{idx}.png"), image_vis[:, :, ::-1])
        # cv2.imwrite(os.path.join(sample_preds_folder, f"org_mask_{idx}.png"), gt_mask[:, :, ::-1])
        cv2.imwrite(os.path.join(sample_preds_folder, f"area_{idx}.png"), pred_mask[:, :, ::-1])
        print(f'saving image mask{idx}...')
    
        #visulaise images
        # visualize(
        #     original_image = image_vis,
        #     # ground_truth_mask = gt_mask,
        #     predicted_mask = pred_mask,
        #     predicted_road_heatmap = pred_road_heatmap
        # )
        
# run_predictions('capture/test/')