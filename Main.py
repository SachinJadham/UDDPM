import torch.utils.data
from torchvision import transforms, utils
from Dataset import TensorDataset, Rescale, ToTensor, RandomResizedCrop, NormalizeImage
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Diffusion import Diffusion
from tqdm import tqdm

from Model import Unet

import torch.optim as optim
import numpy as np
import os

# from torch.utils.tensorboard import SummaryWriter

from torchvision.utils import make_grid

# ============================================ HyperParameters ==================================== #
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# IMAGES_DIR = r'C:\Users\yaa5kor\tensorflow_datasets\FLowerIMages'
IMAGES_DIR = r'/home/i_sjadham77/UUDPM/streetview3k/'
IMAGE_SIZE = 64
BATCH_SIZE = 12
EPOCHS = 500

NOISE_STEPS = 1000
LEARNING_RATE = 0.0001

SAVED_CHECKPOINT_PATH = r'checkpoint/TrainingX'
SAVED_CHECKPOINT_NAME = r'ddpm_ckpt_64_12_after94Epochs.pth'

CHECKPOINT_SAVE_PATH = r'checkpoint/Training1'


def get_data(root_dir=IMAGES_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    # ========================================== TensorDataset ======================================== #
    train_ds = TensorDataset(root_dir,
                                   transform_list=transforms.Compose([Rescale(image_size),
                                                                      RandomResizedCrop(image_size),
                                                                      NormalizeImage(mean=0.5, std=0.5),
                                                                      ToTensor()
                                                                      ]))

    # ========================================== DataLoader =========================================== #
    # create a 'DataLoader'; for managing batches. Each iteration would return a batch.
    trainDL = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # #----- Display a batch ----- #
    # for i_batch, sample_batched in enumerate(trainDL):
    #     print(i_batch, sample_batched.size())
    #     if i_batch == 1:    # display the second batch
    #         plt.figure()
    #         images_batch = sample_batched
    #         # im_size = images_batch.size(2)
    #         grid = utils.make_grid(images_batch)
    #         plt.imshow(grid.numpy().transpose((1, 2, 0)))   # convert to numpy and C x H x W  -> H x W x C
    #         plt.show()
    #         break

    return trainDL


# =================================== LOAD DATA : get DataLoader ==================================== #
train_dl = get_data(IMAGES_DIR, IMAGE_SIZE, BATCH_SIZE)

# ====================================== Forward process ========================================= #
diff = Diffusion(noise_steps=NOISE_STEPS, img_size=IMAGE_SIZE, device=DEVICE)

# ==================================== Initialize Network ========================================== #
# create instance of Network class with constructor arguments
model = Unet(imagechannels=3).to(DEVICE)
# set to train mode
model.train()

# ==================================== Loss and Optimizer ========================================== #
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()

# ================================ Load Checkpoint (if found) ====================================== #
# Load checkpoint if found
if os.path.exists(os.path.join(SAVED_CHECKPOINT_PATH, SAVED_CHECKPOINT_NAME)):
    print('Loading checkpoint as found one')
    model.load_state_dict(torch.load(os.path.join(SAVED_CHECKPOINT_PATH, SAVED_CHECKPOINT_NAME), map_location=DEVICE))

# ==================================== Training ========================================== #
TRAINING = True

if TRAINING:
    # writer = SummaryWriter('runs/Training')  # TensorBoard Writer

    # Create output directories
    if not os.path.exists(CHECKPOINT_SAVE_PATH):
        os.mkdir(CHECKPOINT_SAVE_PATH)

    for epoch_i in range(EPOCHS):
        losses = []
        for i, batch in enumerate(tqdm(train_dl)):
            # move data tensors to CUDA memory
            batch = batch.float().to(DEVICE)

            # sample random noise and timesteps for the batch
            noisy_images_batch, noise, t = diff.forward(batch)

            # display a random image after the forward diffusion of time t
            # diff.display_random_diffused_image(noisy_images_batch, timesteps=t)

            # forward
            noise_pred = model(noisy_images_batch, t)
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())

            # backward
            optimizer.zero_grad()  # set all gradients to zero for each batch
            loss.backward()

            # gradient descent
            optimizer.step()

            if i % 10 == 9:  # every 10 mini-batches.
                writer.add_scalar('training loss', np.mean(losses),
                                  epoch_i * len(train_dl) + i)

        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_i + 1,
            np.mean(losses),
        ))
        torch.save(model.state_dict(),
                   os.path.join(CHECKPOINT_SAVE_PATH,
                                'ddpm_ckpt_{}_{}_after{}Epochs.pth'.format(IMAGE_SIZE, BATCH_SIZE, epoch_i+1)))
print('Training Finished')

# ============================== Sampling/Inference ==================================== #
model.eval()  # set model to eval mode
SAMPLES = 5
with torch.no_grad():
    xt = torch.randn((SAMPLES, 3, IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE)  # [N, C, H, W]
    print('sampling 5 images...')

    for i in tqdm(reversed(range(NOISE_STEPS))):  # i is current timestep
        # get noise prediction from model
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(DEVICE))

        # get xt-1 and x0
        xt, x0_pred = diff.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(DEVICE))

        # # Save x0
        # ims = torch.clamp(xt, -1., 1.).detach().cpu()
        # ims = (ims + 1) / 2
        # grid = make_grid(ims, nrow=1)
        # img = transforms.ToPILImage()(grid)

        #========================== Save x0: custom ==================================#
        import cv2
        from PIL import Image
        PIL_images_list = []
        for img_sample_i in range(SAMPLES):
            img = xt[img_sample_i]      # [1, 3, 416, 416]
            img = torch.squeeze(img)    # [3, 416, 416]
            img = torch.clamp(img, -1., 1.).detach().cpu()
            img = img.numpy().transpose((1, 2, 0))    # C x H x W  -> H x W x C
            img = cv2.normalize(img, None, alpha=0.001, beta=1, norm_type=cv2.NORM_MINMAX)  # [0, 1] range
            img = (img * 255).astype(np.uint8)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            PIL_images_list.append(img)

        def image_grid(imgs, rows, cols):
            assert len(imgs) == rows * cols

            w, h = imgs[0].size
            grid = Image.new('RGB', size=(cols * w, rows * h))
            grid_w, grid_h = grid.size

            for i, img in enumerate(imgs):
                grid.paste(img, box=(i % cols * w, i // cols * h))
            return grid

        img = image_grid(PIL_images_list, rows=1, cols=SAMPLES)
        # ========================== Save x0: custom ==================================#

        if not os.path.exists(os.path.join('checkpoint', 'samples')):
            os.mkdir(os.path.join('checkpoint', 'samples'))
        # img.save(os.path.join('checkpoint', 'samples', 'x0_{}.png'.format(i)))
        if i % 50 == 0:
            img.save(os.path.join('checkpoint', 'samples', 'x0_{}.png'.format(i)))
        img.close()
