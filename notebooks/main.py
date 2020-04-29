import more_itertools as mit
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from box import Box
from lgblkb_tools import logger
from lgblkb_tools.visualize import Plotter
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, random_split, DataLoader

from models.lgblkb_model import TheModel
from src import data_folder
from src.utils import make_train_step
import imgaug.augmenters as iaa

is_cuda_available = torch.cuda.is_available()
logger.info('is_cuda_available: %s', is_cuda_available)
if not is_cuda_available:
    raise SystemError
device = 'cuda' if is_cuda_available else 'cpu'

image_size = (32, 32)


class TheDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, item):
        return self.x[item], self.y[item]
    
    def __len__(self):
        return len(self.y)


def create_data():
    train_df = pd.read_csv(data_folder['raw']['bda-image-challenge-train.txt'], header=None)
    images = train_df.values.reshape((-1, *image_size))
    
    mask_image = np.zeros(image_size)
    mask_image[8:24, 8:24] = 1
    
    # data_shape = (-1, np.product(image_size))
    x = (images * (1 - mask_image) + mask_image)  # .reshape(data_shape)
    y = images  # .reshape(data_shape)
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()
    data = TheDataset(x_tensor, y_tensor)
    return data


def aug_sequencer(images, seed):
    return iaa.Sequential(
        [iaa.Rot90((0, 3), keep_size=False, seed=seed),
         iaa.Fliplr(0.5, seed=seed),
         iaa.Flipud(0.5, seed=seed),
         # iaa.GaussianBlur(),
         ],
        random_order=True,
        seed=seed
    )(images=images)


def augment_batch(batch, seed):
    batch = aug_sequencer(batch.data.numpy().reshape(-1, *image_size), seed=seed)
    batch = np.expand_dims(np.stack(batch), axis=1)
    batch = torch.from_numpy(batch)
    return batch


@logger.trace()
def train():
    torch.manual_seed(369)
    
    dataset = create_data()
    
    train_val_fractions = [0.8, 0.2]
    lenghts = [int(np.round(len(dataset) * fraction)) for fraction in train_val_fractions]
    train_dataset, val_dataset = random_split(dataset, lenghts)
    
    train_batch_size = int(len(train_dataset) / 5)
    logger.info("train_batch_size: %s", train_batch_size)
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True, pin_memory=True)
    wandb.init(project="bda_project")
    
    model = TheModel().to(device)
    # model.load_state_dict(torch.load(model_state_savepath))
    
    wandb.watch(model)
    
    learning_rate = 1e-3
    loss_fn = nn.MSELoss(reduction='sum')
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    
    train_step = make_train_step(model, loss_fn, optimizer)
    
    for epoch in range(200):
        training_losses = list()
        for x_batch_init, y_batch_init in train_loader:
            # for pair in zip(x_batch, y_batch):
            #     Plotter(*pair)
            
            # raise NotImplementedError
            for batch_idx in range(8):
                seed = np.random.randint(0, 100000000)
                x_batch = augment_batch(x_batch_init, seed)
                y_batch = augment_batch(y_batch_init, seed)
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                training_loss = train_step(x_batch, y_batch)
                training_losses.append(training_loss)
        train_loss_average = np.mean(training_losses) / train_batch_size
        wandb.log({"Training loss (average)": train_loss_average})
        
        if epoch % 20 == 0:
            scheduler.step(train_loss_average)
            val_losses = list()
            model.eval()
            with torch.no_grad():
                worst_example = Box()
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    
                    yhat_val = model(x_val)
                    val_loss = loss_fn(y_val, yhat_val).item()
                    val_losses.append(val_loss)
                    if worst_example.get('val_loss', 0) > val_loss: continue
                    
                    worst_example.x_image = x_val.detach().data.reshape(image_size)
                    worst_example.y_image = y_val.detach().data.reshape(image_size)
                    worst_example.yhat_image = yhat_val.detach().data.reshape(image_size)
                    worst_example.val_loss = val_loss
                
                images = worst_example.x_image, worst_example.yhat_image, worst_example.y_image
                wandb.log({f"Epoch {epoch} worst": [wandb.Image(i) for i in images]})
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'model_epoch_{epoch}.pt'))
            model.train()
            val_loss_average = np.mean(val_losses)
            wandb.log({"Validation Loss": val_loss_average})
    # torch.save(model.state_dict(), model_state_savepath)
    
    # plt.plot(losses, label='Training loss')
    # plt.plot(val_losses, label='Validation loss')
    # plt.legend()
    # plt.show()
    #
    pass


def test():
    torch.manual_seed(369)
    model = TheModel()
    state_dict_path = '/home/lgblkb/PycharmProjects/abda_project/wandb/run-20200426_113911-wxzvb2i8/model.pt'
    model.load_state_dict(torch.load(state_dict_path))
    
    dataset = create_data()
    
    train_val_fractions = [0.8, 0.2]
    lenghts = [int(np.round(len(dataset) * fraction)) for fraction in train_val_fractions]
    train_dataset, val_dataset = random_split(dataset, lenghts)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)
    
    plotter = Plotter()
    for i, (x, y) in enumerate(val_loader):
        if i % 5 == 0:
            plotter.plot(rows_cols=(5, 3))
            plotter = Plotter()
        
        yhat = model(x).data.reshape((32, 32))
        x = x.data.reshape((32, 32))
        y = y.data.reshape((32, 32))
        plotter.add_images(x, y, yhat)
    
    pass


def main():
    train()
    pass


if __name__ == '__main__':
    main()
