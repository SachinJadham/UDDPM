import torch
import matplotlib.pyplot as plt
import cv2

# plt.ion()  # interactive mode for plots

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=416, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.betas = torch.linspace(self.beta_start, self.beta_end, self.noise_steps).to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

        # needed for reverse process
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)

    def add_noise(self, batch, noise, t):

        # #x_shape = batch.shape  # [B, C, H, W]
        # sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]  # reshape to tensor of size [B 1 1 1]
        # sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        #
        # return sqrt_alpha_hat * batch + sqrt_one_minus_alpha_hat * noise

        original_shape = batch.shape
        batch_size = original_shape[0]

        sqrt_alpha_hat = self.sqrt_alpha_hat.to(batch.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat.to(batch.device)[t].reshape(batch_size)

        # Reshape till (B, ) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_hat = sqrt_alpha_hat.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat.unsqueeze(-1)

        # Apply and Return Forward process equation
        return (sqrt_alpha_hat.to(batch.device) * batch
                + sqrt_one_minus_alpha_hat.to(batch.device) * noise)

    def forward(self, batch):
        # sample Random Noise
        noise = torch.randn_like(batch).to(self.device)  # [B, C, H, W]

        # sample time step b/w [0:NOISE_STEPS] of size B
        t = torch.randint(low=1, high=self.noise_steps, size=(batch.shape[0],)).to(self.device)  # [B, ]

        noised_batch = self.add_noise(batch, noise, t)

        return noised_batch, noise, t

    def display_random_diffused_image(self, fwd_batch, timesteps):
        r""" to be called after calling the forward function"""
        # random number b/w [0, B]
        img_idx = torch.randint(0, fwd_batch.shape[0], (1,)).to(self.device)
        target_img = fwd_batch[img_idx]     # [1, 3, 416, 416]
        target_img = torch.squeeze(target_img)  # [3, 416, 416]

        target_img = target_img.to('cpu').numpy().transpose((1, 2, 0))    # C x H x W  -> H x W x C
        timestep_for_image = timesteps.to('cpu').numpy()[img_idx]

        target_img = cv2.normalize(target_img, None, alpha=0.001, beta=1, norm_type=cv2.NORM_MINMAX)  # [0, 1] range

        plt.imshow(target_img)
        plt.title(f'timestep of Noise in this image: {timestep_for_image}')
        plt.show()
        print(target_img.max())     # after Noise addition the max is > 1.0; Normalization is gone.

    def sample_prev_timestep(self, xt, noise_prediction, t):
        r"""
            Use the noise prediction by model to get
            xt-1 using xt and the noise predicted
        :param xt: current timestep sample
        :param noise_prediction: model noise prediction
        :param t: current timestep we are at
        :return: xt-1, x0
        """
        x0 = ((xt - (self.sqrt_one_minus_alpha_hat.to(xt.device)[t] * noise_prediction)) /
              torch.sqrt(self.alpha_hat.to(xt.device)[t]))
        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - (
                (self.betas.to(xt.device)[t]) * noise_prediction) / (self.sqrt_one_minus_alpha_hat.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])

        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_hat.to(xt.device)[t - 1]) / (1.0 - self.alpha_hat.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)

            # OR
            # variance = self.betas[t]
            # sigma = variance ** 0.5
            # z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0
