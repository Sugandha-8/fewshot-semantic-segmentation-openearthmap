import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def load_images_and_masks(img_path, mask_path, start_index=0, end_index=None):
    img_files = sorted(os.listdir(img_path))[start_index:end_index]
    mask_files = sorted(os.listdir(mask_path))

    mask_dict = {os.path.splitext(mask)[0]: mask for mask in mask_files}

    image_list, mask_list = [], []
    for img_file in img_files:
        img_base = os.path.splitext(img_file)[0]
        img_full_path = os.path.join(img_path, img_file)
        try:
            image = Image.open(img_full_path).convert('RGB')
        except IOError:
            print(f"Failed to load image: {img_file}")
            continue

        

        if img_base in mask_dict:
            mask_full_path = os.path.join(mask_path, mask_dict[img_base])
            try:
                mask = Image.open(mask_full_path).convert('L')  # Convert to grayscale
            except IOError:
                print(f"Failed to load mask: {mask_full_path}")
                mask = None  
        else:
            mask = None  

       

        image_list.append(np.array(image))
        mask_list.append(np.array(mask) if mask is not None else None)

    return image_list, mask_list



train_image_folder = "/scratch/r.sugandha/oem/oem_data/trainset/images/"
train_mask_folder = "/scratch/r.sugandha/oem/oem_data/trainset/labels/"
val_image_folder = "/scratch/r.sugandha/oem/oem_data/valset/images/"
val_mask_folder = "/scratch/r.sugandha/oem/oem_data/valset/labels/"

train_images, train_masks = load_images_and_masks(train_image_folder, train_mask_folder, 0, 200)
pseudo_test_images, pseudo_test_masks = load_images_and_masks(train_image_folder, train_mask_folder, 200, 258)


val_images, val_masks = load_images_and_masks(val_image_folder, val_mask_folder)




def check_data(images, masks, expected_num, label="Training"):
    print(f"--- {label} Data Check ---")
    if len(images) != expected_num:
        print(f"Error: Expected {expected_num} images, but got {len(images)}.")
    else:
        print(f"Number of images: {len(images)}")

    if masks:
        if len(images) == len(masks):
            print(f"Number of masks matches the number of images.")
        else:
            print(f"Error: Number of images and masks do not match.")
    else:
        print("No masks to check.")

    
    if images:
        print(f"Sample image shape: {images[0].shape}")
        print(f"Sample image dtype: {images[0].dtype}")
    if masks and masks[0] is not None:
        print(f"Sample mask shape: {masks[0].shape}")
        print(f"Sample mask dtype: {masks[0].dtype}")

def visualize_sample(images, masks, num_samples=5):
    plt.figure(figsize=(15, 5))
    for i in range(min(num_samples, 6)):
        plt.subplot(2, num_samples, i+1)
        plt.imshow(images[i])
        plt.title('Image')
        plt.axis('off')
        
        if masks and masks[i] is not None:
            plt.subplot(2, num_samples, num_samples+i+1)
            plt.imshow(masks[i], cmap='gray')
            plt.title('Mask')
            plt.axis('off')
    plt.show()
    plt.savefig("vis_training.png")

 #check_data(train_images, train_masks, 200, "Training")
 # print("Visualizing Training Data Samples:")
 # visualize_sample(train_images, train_masks)






class CustomDataset(Dataset):
    def __init__(self, images, masks=None, image_transform=None, mask_transform=None):
        self.images = images
        self.masks = masks
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
       
        image = self.images[idx]
        mask = self.masks[idx] if self.masks is not None and self.masks[idx] is not None else None

      
        image = transforms.functional.to_pil_image(image)
        if mask is not None:
            mask = transforms.functional.to_pil_image(mask)

        
        if self.image_transform:
            image = self.image_transform(image)
        if mask is not None and self.mask_transform:
            mask = self.mask_transform(mask)
            mask = torch.squeeze(mask, 0)
            mask = mask.long()

        return image, mask


image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])





train_dataset = CustomDataset(train_images, train_masks, image_transform=image_transform, mask_transform=mask_transform)
val_dataset = CustomDataset(val_images, val_masks, image_transform=image_transform, mask_transform=mask_transform)




train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)




def display_sample(dataset):
    image, mask = dataset[0]  # Get the first sample from the dataset
    image = image.permute(1, 2, 0)  # Rearrange channels for plotting
    plt.imshow(image.numpy())
    plt.title('Sample Image After Resizing')
    plt.axis('off')
    plt.show()
    plt.savefig("resized.png")





train_dataset = CustomDataset(train_images, train_masks, image_transform=image_transform, mask_transform=mask_transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)


val_dataset = CustomDataset(val_images, val_masks, image_transform=image_transform, mask_transform=mask_transform)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

def check_class_distribution(masks):
    class_counts = np.unique(np.concatenate([mask.flatten() for mask in masks if mask is not None]), return_counts=True)
    print("MASK DIST", dict(zip(*class_counts)))

check_class_distribution(train_masks)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ConvBlock, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

      
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        skip = self.skip(x)
        x = self.conv1(x)
        print("shape after conv1 ", x.shape)
        x = self.bn1(x)
        x = self.relu1(x)
        print("shape after conv1 relu", x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        print("shape after conv2 relu", x.shape)
        x = self.conv3(x)
        x = self.bn3(x)
        x += skip
        x = F.relu(x)
        print("shape after conv3 ", x.shape)
        return x

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 4, 8]):
        super(PyramidPoolingModule, self).__init__()
        self.pools = []
        self.in_channels = in_channels
        self.pool_sizes = pool_sizes
        for size in pool_sizes:
            self.pools.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(size, size)),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            ))
        self.pools = nn.ModuleList(self.pools)

    def forward(self, x):
        input_size = x.shape[2:]  # Capture spatial dimensions
        features = [x]
        print(f"Input feature map shape: {x.shape}")
        for pool in self.pools:
            pooled = pool(x)
            print(f"Pooled shape ({pool[0].output_size}): {pooled.shape}")
            upsampled = F.interpolate(pooled, size=input_size, mode='bilinear', align_corners=False)
            print(f"Upsampled shape: {upsampled.shape}")
            features.append(upsampled)
        result = torch.cat(features, dim=1)
        print(f"Concatenated result shape: {result.shape}")
        return result





class SegNet(nn.Module):
    def __init__(self, num_classes):
        super(SegNet, self).__init__()
        self.layer1 = ConvBlock(3, 64)
        self.layer2 = ConvBlock(64, 128)
        self.layer3 = ConvBlock(128, 256)
        self.pool = PyramidPoolingModule(256)
        self.final = nn.Conv2d(256 * 2, num_classes, kernel_size=1)  # No reduction in the number of features

    def forward(self, x):
        x = self.layer1(x)
        print("final layer1 shape ", x.shape)
        x = self.layer2(x)
        print("final layer2 shape ", x.shape)
        x = self.layer3(x)
        print("final layer3 shape ", x.shape)
        x = self.pool(x)
        print("final pool x shape ", x.shape)
        pool_output = x 
        x = self.final(x)
        print("final x shape ", x.shape)
        return x, pool_output





def visualize_meanfeature_maps(outputs, save_path=None):
    if outputs.dim() == 4:  
      
        print("mean feature func")
        print("meanfeatures func's input shape",outputs.shape)
        output_mean = outputs.mean(dim=1).detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 10))
        # Visualize the first example in the batch
        ax.imshow(output_mean[1], cmap='hot')
        ax.axis('off')
        ax.set_title('Mean Feature Map')

        if save_path:
            plt.savefig(save_path)
        plt.show()
        plt.savefig("mean_feature2.png")

    else:
        print("Invalid input dimensions for visualization.")

def visualize_specific_channels(outputs, num_samples, num_channels_to_display, save_path=None):
    for sample_idx in range(num_samples):  # Iterate through each sample
        sample_output = outputs[sample_idx]  # Select the specific sample's output

        num_channels = min(num_channels_to_display, sample_output.shape[0])
        fig, axes = plt.subplots(1, num_channels, figsize=(20, 2))

        if num_channels == 1:
            axes = [axes]

        for channel_idx in range(num_channels):
            print("NEW FUNC")
            channel_img = sample_output[channel_idx].detach().cpu().numpy()
            axes[channel_idx].imshow(channel_img, cmap='viridis')
            axes[channel_idx].axis('off')
            axes[channel_idx].set_title(f'Sample {sample_idx} Channel {channel_idx}')

        plt.show()
        plt.savefig('sampleChannel.png')
        if save_path:
            plt.savefig(os.path.join(save_path, f"sample_{sample_idx}_channels.png"))
        plt.close()  # Make sure to close the plot


# def visualize_specific_channels(outputs, num_channels_to_display, save_path=None):
#     if outputs.dim() == 4:  
       
#         outputs = outputs[0]
        
       
#         num_channels = min(num_channels_to_display, outputs.shape[0])
        
  
#         fig, axes = plt.subplots(1, num_channels, figsize=(20, 2))
        
      
#         if num_channels == 1:
#             axes = [axes]
#         print("inside channel func")
#         for i in range(num_channels):
#             channel_img = outputs[i].detach().cpu().numpy()
#             print("inside channel func-2")
#             print(num_channels)

           
            
#             axes[i].imshow(channel_img, cmap='viridis')
#             axes[i].axis('off')
#             axes[i].set_title(f'Channel {i}')
            
           
#             if save_path:
#                 plt.savefig(os.path.join(save_path, f"channel_{i}.png"))

#         plt.show()
#         plt.savefig("channel.png")
       
#         if save_path:
#             plt.savefig(os.path.join(save_path, "specific_channels.png"))

#     else:
#         print("Invalid input dimensions for specific channel visualization.")
    


### save pred section starts
def check_logits_and_predictions(loader, model, device="cuda"):
    model.eval()  
    examples = next(iter(loader)) 
    inputs, _ = examples
    inputs = inputs.to(device)

    with torch.no_grad(): 
        outputs, _ = model(inputs) 

      
        print("Sample logits:", outputs[0])
        print("Sample logits shape:", outputs[0].shape)

        probs = F.softmax(outputs, dim=1)
        print("Sample probabilities:", probs[0])

        
        preds = torch.argmax(probs, dim=1)  # Convert probabilities to class predictions
        print("Sample predictions:", preds[0])
        print("Unique predicted classes in the batch:", torch.unique(preds))

    return preds






colors = np.array([
    [255, 0, 0],    # Red
    [0, 255, 0],    # Green
    [0, 0, 255],    # Blue
    [255, 255, 0],  # Yellow
    [255, 0, 255],  # Magenta
    [0, 255, 255],  # Cyan
    [128, 128, 128] # Gray
], dtype=np.uint8)

def label_to_rgb(mask):
    
    mask = np.clip(mask, 0, len(colors) - 1)
    print("mask shape",mask.shape)
    rgb_image = colors[mask]
    
    return rgb_image

def save_and_plot_predictions(loader, model, folder="predictions/", device="cuda"):
    model.eval()
    os.makedirs(folder, exist_ok=True)

    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds, _ = model(x)
            preds = torch.argmax(preds, dim=1)  # Get class indices

            # Print the unique predicted classes to the console.
            unique, counts = np.unique(preds.cpu().numpy(), return_counts=True)
            print(f"Batch {idx} - Unique classes and counts:", dict(zip(unique, counts)))

        # Save and plot only the predictions, not the channels.
        for i, pred in enumerate(preds):
            pred_rgb = label_to_rgb(pred.cpu().numpy())
            plt.figure(figsize=(3, 3))  # Adjust the figure size if needed
            plt.imshow(pred_rgb)
            plt.axis('off')
            plt.title(f'Prediction {idx * loader.batch_size + i}')
            plt.savefig(os.path.join(folder, f"prediction_{idx * loader.batch_size + i}.png"))
            plt.close()  # Close the plot to avoid displaying it inline if not required

        if idx == 5:  # Stop after a certain number of batches to avoid generating too many images.
            break

# def save_and_plot_predictions(loader, model, folder="predictions/", device="cuda"):
#     model.eval()
#     os.makedirs(folder, exist_ok=True)

#     for idx, (x, y) in enumerate(loader):
#         x = x.to(device)
#         with torch.no_grad():
#             preds, _ = model(x)
#             preds = torch.argmax(preds, dim=1)  # Get class indices
#             unique, counts = np.unique(preds.cpu().numpy(), return_counts=True)
#             print("Unique classes and counts:", dict(zip(unique, counts)))
#             preds_rgb = [label_to_rgb(pred.cpu().numpy()) for pred in preds]

#         # Save each prediction as an image
#         for i, pred_rgb in enumerate(preds_rgb):
#             plt.imshow(pred_rgb)
#             plt.axis('off')
#             plt.title(f'Prediction {idx*loader.batch_size + i}')
#             plt.savefig(os.path.join(folder, f"pred_{idx*loader.batch_size + i}.png"))
#             plt.show()

#         if idx == 5:  # Limit to first 6 batches to avoid too many outputs
#             break



### save pred section ends



def train_model(model, dataloaders, criterion, optimizer, num_epochs=2, device='cuda', save_path='model_weights.pth'):
    model.to(device)
    best_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                print("Training mode")
            else:
                model.eval()
                print("Evaluation mode")

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    predictions, pool_outputs = model(inputs)
                    loss = criterion(predictions, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), save_path)
                print(f'Model saved at {save_path} with loss {best_loss:.4f}')
            
            ### new section for plotting prediction map
            if not os.path.exists('predictions'):
                os.makedirs('predictions')
            if phase == 'train':
                save_and_plot_predictions(dataloaders['train'], model, folder="predictions/", device=device)
                check_logits_and_predictions(dataloaders['train'], model, device=device)
            ### new section ends

            # Visualization at the end of validation phase
            if phase == 'val':
                # Getting the last batch from the validation set
                inputs, _ = next(iter(dataloaders['val']))
                inputs = inputs.to(device)
                with torch.no_grad():
                    model.eval()
                    _, pool_outputs = model(inputs)
               #     visualize_feature_maps(pool_outputs, save_path=f"pyramid_pooling_maps_epoch_{epoch}.png")
                  #  print("one function's done")
                    visualize_meanfeature_maps(pool_outputs)
                    print("mean feature function's done")
                    # uncomment the following 
                    visualize_specific_channels(pool_outputs,num_samples=2,num_channels_to_display=8)
                    print("specific channel function's done")



    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('loss300.png')

    return model







model = SegNet(num_classes=8)  
#total_pixels = np.sum([55570599, 35834573, 39187255, 1490587, 24941547, 13608815, 2470844, 26756444])
class_counts = np.array([55570599, 35834573, 39187255, 1490587, 24941547, 13608815, 2470844, 26756444])


total_pixels = np.sum(class_counts)


class_weights = total_pixels / class_counts

class_weights = class_weights / np.min(class_weights)


class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device="cuda")

# Initialize the loss function with these weights
criterion = nn.CrossEntropyLoss(weight=class_weights)

#class_weights = torch.tensor([total_pixels/count for count in [55570599, 35834573, 39187255, 1490587, 24941547, 13608815, 2470844, 26756444]], dtype=torch.float32).to(device)

#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


dataloaders = {
    'train': train_loader,
    'val': val_loader
}


trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=300, save_path='bestEP300_model_weights.pth')
