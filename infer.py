# %%
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.optim as optim
import os
import h5py
from PIL import Image
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset, DataLoader
dir = '/kaggle/input/a2data/'
out_dir = '/kaggle/working/'
vae_checkpoint_path = './vae_checkpoint.pth'
batch_size = 1

class ARGS:
    def __init__(self):
        self.input_dir = './Samples'
        self.model_path = './part2_model.pth'
        self.part = 2
        self.output_dir = './out'

args = ARGS()

# %%
def preprocess(path, resize_shape, center_crop_shape, to_crop = True):
    data = []
    for f in sorted(os.listdir(path)):
        img = Image.open(path + f)
        if(to_crop):
            img.crop((center_crop_shape[0], center_crop_shape[2], center_crop_shape[1], center_crop_shape[3]))
        img = img.resize((resize_shape[1], resize_shape[2]), )
        img = np.array(img)
        if len(img.shape) == 3:
            img = img[:, :, :3]
            img = torch.tensor(img).permute(2, 0, 1)
        else:
            img = torch.tensor(img)
        data.append(img)

    return data
    
def make_h5py(fname, path):
    os.system('rm -f ' + fname)
    with h5py.File(fname, 'w') as hdf5_file:
        images = preprocess(path, (3, 128, 128), (-1, -1, -1, -1), False)
        hdf5_file.create_dataset('images', shape = (len(images), 3, 128, 128), data = images)


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        
        # Open HDF5 file
        self.hdf5_handle = h5py.File(hdf5_file, 'r')
        self.data = self.hdf5_handle['images']
        self.images = [torch.tensor(self.data[i], dtype=torch.float32) / 255 for i in range(len(self.data))]
        # self.masks = self.hdf5_handle['masks']
        # self.masks = [torch.tensor(self.masks[i], dtype=torch.uint8) for i in range(len(self.masks))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]

# %%
class SlotAttention(nn.Module):
    def __init__(self, k, d_common=64, n_iter_train=3,n_iter_test=5, d_slot=64, d_inputs=64, hid_dim=128):
        super(SlotAttention, self).__init__()
        self.k = k
        self.d_common = d_common
        self.n_iter_train = n_iter_train
        self.n_iter_test = n_iter_test
        self.d_slot = d_slot
        self.d_inputs = d_inputs

        self.fc_q = nn.Linear(d_slot, d_common)
        self.fc_k = nn.Linear(d_inputs, d_common)
        self.fc_v = nn.Linear(d_inputs, d_common)

        self.gru = nn.GRUCell(d_common, d_slot)
        self.mlp = nn.Sequential(
            nn.Linear(d_slot, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, d_slot)
        )

        self.softmax = nn.Softmax(dim=2)
        self.mu = nn.Parameter(torch.randn(1, 1,d_common))
        self.sigma = nn.Parameter(torch.rand(1,1, d_common))

    
    def forward(self, inputs):
        # inputs: (batch_size, n_inputs, d_inputs)
        # slots: (batch_size, n_slots, d_slot)
        if self.training:
            n_iter = self.n_iter_train
        else:
            n_iter = self.n_iter_test
        batch_size, n_inputs, d_inputs = inputs.size()
        mu = self.mu.expand(batch_size, self.k, -1)
        sigma = self.sigma.expand(batch_size, self.k, -1)
        slots = torch.normal(mu, sigma).to(device)
        inputs = nn.LayerNorm(d_inputs).to(device)(inputs)
        k = self.fc_k(inputs)               # (batch_size, n_inputs, d_common)
        v = self.fc_v(inputs)               # (batch_size, n_inputs, d_common)
        for i in range(n_iter):
            q = self.fc_q(nn.LayerNorm(self.d_slot).to(device)(slots))                # (batch_size, n_slots, d_common)

            attn = torch.bmm(k, q.permute(0, 2, 1)) / np.sqrt(self.d_common)            # (batch_size, n_inputs, n_slots)
            attn = self.softmax(attn) +  1e-8                                           # (batch_size, n_inputs, n_slots)
            attn = attn / attn.sum(dim=1, keepdim=True)                                 # (batch_size, n_inputs, n_slots)
            attn = attn.permute(0,2,1)
            updates = torch.einsum('bjd,bij->bid', v, attn)                             # (batch_size, n_slots, d_common)


            slots = self.gru(updates.reshape(-1,self.d_common), slots.reshape(-1, self.d_slot)).reshape(batch_size, self.k, self.d_slot)
            slots = nn.LayerNorm(self.d_slot).to(device)(slots)
            slots = slots + self.mlp(slots)
        
        return slots

class PositionalEmbeddings(nn.Module):
    def __init__(self, H, W, hid_dim=64):
        super(PositionalEmbeddings, self).__init__()
        self.H = H
        self.W = W
        self.hid_dim = hid_dim
        self.project = nn.Linear(4, hid_dim)
    
    def construct_grid(self, H, W):
        x = torch.linspace(0, 1, W).unsqueeze(0).repeat(H, 1)
        y = torch.linspace(0, 1, H).unsqueeze(1).repeat(1, W)
        return torch.stack([x, 1-x, y, 1-y], dim=2)    # (H, W, 4)


    def forward(self, inputs):
        grid = self.construct_grid(self.H, self.W).to(device)  # (H, W, 4)
        grid = self.project(grid)
        return inputs + grid.unsqueeze(0).expand(inputs.size(0), self.H, self.W, self.hid_dim)
    

class CNNEncoder(nn.Module):
    def __init__(self, hid_dim=64):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, hid_dim, 5, padding=2)                    
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)

        self.positionalEmb = PositionalEmbeddings(128, 128, hid_dim)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)  


    def forward(self, inputs):
        inputs = self.conv1(inputs)
        inputs = self.relu(inputs)
        inputs = self.conv2(inputs)
        inputs = self.relu(inputs)
        inputs = self.conv3(inputs)
        inputs = self.relu(inputs)
        inputs = self.conv4(inputs)
        inputs = self.relu(inputs)
        
        inputs = self.positionalEmb(inputs.permute(0, 2, 3, 1))
        inputs = inputs.flatten(1, 2)
        inputs = nn.LayerNorm(inputs.size()[1:]).to(device)(inputs) 
        inputs = self.fc1(inputs)
        inputs = self.relu(inputs)
        inputs = self.fc2(inputs)
        return inputs
           

class deconvDecoder(nn.Module):
    def __init__(self, hid_dim=64):
        super(deconvDecoder, self).__init__()
        self.hid_dim = hid_dim
        self.deconv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, padding=2, output_padding=1, stride=2)
        self.deconv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, padding=2, output_padding=1, stride=2)
        self.deconv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, padding=2, output_padding=1, stride=2)
        self.deconv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, padding=2, output_padding=1, stride=2)
        self.deconv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, padding=2, output_padding=0, stride=1)
        self.deconv6 = nn.ConvTranspose2d(hid_dim, 4, 3, padding=1, output_padding=0, stride=1)

        self.relu = nn.ReLU()
        self.positionalEmb = PositionalEmbeddings(8, 8, hid_dim)

    def forward(self, slots, name, out_dir):
        # slots: (batch_size, n_slots, d_slot)
        b, k, d = slots.size()
        slots = slots.unsqueeze(2).unsqueeze(3).expand(b, k, 8, 8, d)
        slots = slots.reshape(b*k, 8, 8, d)
        slots = self.positionalEmb(slots)
        slots = slots.permute(0, 3, 1, 2)
        slots = self.deconv1(slots)
        slots = self.relu(slots)
        slots = self.deconv2(slots)
        slots = self.relu(slots)
        slots = self.deconv3(slots)
        slots = self.relu(slots)
        slots = self.deconv4(slots)
        slots = self.relu(slots)
        slots = self.deconv5(slots)
        slots = self.relu(slots)
        slots = self.deconv6(slots)                 # (batch_size * n_slots, 4, 128, 128)

        slots = slots.reshape(b, k, 4, 128, 128)
        slots = slots.permute(0, 1, 3, 4, 2)
        contents, masks = slots.split([3, 1], dim=4)        # (batch_size, n_slots, 128, 128, 3)
        masks = nn.Softmax(dim=1)(masks)                    # (batch_size, n_slots, 128, 128, 1)
        img = (contents * masks).sum(dim=1)
        img = img.permute(0, 3, 1, 2)
        img_recon = torch.tensor(((contents * masks).sum(dim=1) * 255), dtype=torch.uint8).cpu().numpy()
        for i in range(b):
#             for j in range(k):
#                 img_name = name + '_slot' + str(j) + '_img' + str(i) + '.png'
#                 plt.imsave(out_dir + img_dir + img_name, img[i,j])
            plt.imsave(out_dir + name + '.png', img_recon[i])

        return img, masks
    
    def get_slot_imgs(self, slots, name, out_dir):
        b, k, d = slots.size()
        slots = slots.unsqueeze(2).unsqueeze(3).expand(b, k, 8, 8, d)
        slots = slots.reshape(b*k, 8, 8, d)
        slots = self.positionalEmb(slots)
        slots = slots.permute(0, 3, 1, 2)
        slots = self.deconv1(slots)
        slots = self.relu(slots)
        slots = self.deconv2(slots)
        slots = self.relu(slots)
        slots = self.deconv3(slots)
        slots = self.relu(slots)
        slots = self.deconv4(slots)
        slots = self.relu(slots)
        slots = self.deconv5(slots)
        slots = self.relu(slots)
        slots = self.deconv6(slots)                 # (batch_size * n_slots, 4, 128, 128)

        slots = slots.reshape(b, k, 4, 128, 128)
        slots = slots.permute(0, 1, 3, 4, 2)
        contents, masks = slots.split([3, 1], dim=4)        # (batch_size, n_slots, 128, 128, 3)
        masks = nn.Softmax(dim=1)(masks)                    # (batch_size, n_slots, 128, 128, 1)
        img = torch.tensor((contents * masks) * 255, dtype=torch.uint8).cpu().numpy()
        img_dir = 'img_dir/'
        if(not os.path.exists(out_dir + img_dir)):
            os.makedirs(out_dir + img_dir)
        img_recon = torch.tensor(((contents * masks).sum(dim=1) * 255), dtype=torch.uint8).cpu().numpy()
        for i in range(b):
#             for j in range(k):
#                 img_name = name + '_slot' + str(j) + '_img' + str(i) + '.png'
#                 plt.imsave(out_dir + img_dir + img_name, img[i,j])
            plt.imsave(out_dir + name + '_' + str(i) + '.png', img_recon[i])
        


class SlotAttentionModel(nn.Module):
    def __init__(self, k, d_common=64, n_iter_train=3, n_iter_test=5, d_slot=64, d_inputs=64, hid_dim=64):
        super(SlotAttentionModel, self).__init__()
        self.encoder = CNNEncoder(hid_dim)
        self.slotAttention = SlotAttention(k, d_common, n_iter_train, n_iter_test, d_slot, d_inputs, hid_dim)
        self.decoder = deconvDecoder(hid_dim)
    
    def forward(self, inputs, name, out_dir):
        features = self.encoder(inputs)
        slots = self.slotAttention(features)
        img, masks = self.decoder(slots, name, out_dir)
        return img, masks
    
    def get_slot_imgs(self, inputs, name, out_dir):
        features = self.encoder(inputs)
        slots = self.slotAttention(features)
        self.decoder.get_slot_imgs(slots, name, out_dir)




# %%
# !pip3 install einops

from einops import rearrange


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self,
        n_e,
        e_dim,
        beta,
        remap=None,
        unknown_index="random",
        sane_index_shape=False,
        legacy=True,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(
                0, self.re_embed,
                size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(
                z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,
                                                                1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):

    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h
    
class PositionalEmbeddings(nn.Module):
    def __init__(self, H, W, hid_dim=64):
        super(PositionalEmbeddings, self).__init__()
        self.H = H
        self.W = W
        self.hid_dim = hid_dim
        self.project = nn.Linear(4, hid_dim)
    
    def construct_grid(self, H, W):
        x = torch.linspace(0, 1, W).unsqueeze(0).repeat(H, 1)
        y = torch.linspace(0, 1, H).unsqueeze(1).repeat(1, W)
        return torch.stack([x, 1-x, y, 1-y], dim=2)    # (H, W, 4)


    def forward(self, inputs):
        grid = self.construct_grid(self.H, self.W).to(device)  # (H, W, 4)
        grid = self.project(grid)
        return inputs + grid.unsqueeze(0).expand(inputs.size(0), self.H, self.W, self.hid_dim)


class AttnBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(
            v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "none"], f'attn_type {attn_type} unknown'
    print(f"making '{attn_type}' attention with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        raise NotImplementedError(f'attn_type {attn_type} not implemented')


class Encoder(nn.Module):

    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=False,
        attn_type="vanilla",
        **ignore_kwargs,
    ):
        super().__init__()

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1, ) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        out_ch = z_channels * 2 if double_z else z_channels
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):

    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        attn_type="vanilla",
        **ignorekwargs,
    ):
        super().__init__()

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1, ) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2**(self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


def temporal_wrapper(func):
    """A wrapper to make the model compatible with both 4D and 5D inputs."""

    def f(cls, x):
        """x is either [B, C, H, W] or [B, T, C, H, W]."""
        B = x.shape[0]
        if len(x.shape) == 5:
            unflatten = True
            x = x.flatten(0, 1)
        else:
            unflatten = False

        outs = func(cls, x)

        if unflatten:
            if isinstance(outs, tuple):
                outs = [o.unflatten(0, (B, -1)) if o.ndim else o for o in outs]
                return tuple(outs)
            else:
                return outs.unflatten(0, (B, -1))
        else:
            return outs

    return f


class VAE(torch.nn.Module):
    """VQ-VAE consisting of Encoder, QuantizationLayer and Decoder."""

    def __init__(
        self,
        enc_dec_dict=dict(
            resolution=128,
            in_channels=3,
            z_channels=3,
            ch=64,  # base_channel
            ch_mult=[1, 2, 4],  # num_down = len(ch_mult)-1
            num_res_blocks=2,
            attn_resolutions=[],
            out_ch=3,
            dropout=0.0,
        ),
        vq_dict=dict(
            n_embed=4096,  # vocab_size
            embed_dim=3,  # same as `z_channels`
            percept_loss_w=1.0,
        ),
        use_loss=True,
    ):
        super().__init__()

        self.resolution = enc_dec_dict['resolution']
        self.embed_dim = vq_dict['embed_dim']
        self.n_embed = vq_dict['n_embed']
        self.z_ch = enc_dec_dict['z_channels']

        self.encoder = Encoder(**enc_dec_dict)
        self.decoder = Decoder(**enc_dec_dict)

        self.quantize = VectorQuantizer(
            self.n_embed,
            self.embed_dim,
            beta=0.25,
            sane_index_shape=True,
        )
        self.quant_conv = nn.Conv2d(self.z_ch, self.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(self.embed_dim, self.z_ch, 1)

        # if use_loss:
        #     self.loss = VQLPIPSLoss(percept_loss_w=vq_dict['percept_loss_w'])

    @temporal_wrapper
    def encode_quantize(self, x):
        """Encode image to pre-VQ features, then quantize."""
        h = self.encode(x)  # `embed_dim`
        quant, quant_loss, (_, _, quant_idx) = self.quantize(h)
        # [B, `embed_dim`, h, w], scalar, [B*h*w]
        return quant, quant_loss, quant_idx

    @temporal_wrapper
    def encode(self, x):
        """Encode image to pre-VQ features."""
        # this is the x0 in LDM!
        h = self.encoder(x)  # `z_ch`
        h = self.quant_conv(h)  # `embed_dim`
        return h
    
    def decode(self, latent):
        quant, quant_loss, (_, _, quant_idx) = self.quantize(latent)
        image = self._decode(quant)
        return image

    @temporal_wrapper
    def quantize_decode(self, h):
        """Input pre-VQ features, quantize and decode to reconstruct."""
        # use this to reconstruct images from LDM's denoised output!
        quant, _, _ = self.quantize(h)
        dec = self.decode(quant)
        return dec

    @temporal_wrapper
    def _decode(self, quant):
        """Input already quantized features, do reconstruction."""
        quant = self.post_quant_conv(quant)  # `z_ch`
        dec = self.decoder(quant)
        return dec

    def forward(self, data_dict):
        img = data_dict['img']
        quant, quant_loss, token_id = self.encode_quantize(img)
        recon = self.decode(quant)
        out_dict = {
            'recon': recon,
            'token_id': token_id,
            'quant_loss': quant_loss,
        }
        return out_dict

    def calc_train_loss(self, data_dict, out_dict):
        """Compute training loss."""
        img = data_dict['img']
        recon = out_dict['recon']
        quant_loss = out_dict['quant_loss']

        loss_dict = self.loss(quant_loss, img, recon)

        return loss_dict

    @torch.no_grad()
    def calc_eval_loss(self, data_dict, out_dict):
        """Loss computation in eval."""
        loss_dict = self.calc_train_loss(data_dict, out_dict)
        img = data_dict['img']
        recon = out_dict['recon']
        loss_dict['recon_mse'] = F.mse_loss(recon, img)
        return loss_dict

    @property
    def dtype(self):
        return self.quant_conv.weight.dtype

    @property
    def device(self):
        return self.quant_conv.weight.device



# %%
class SlotAttention2(nn.Module):
    def __init__(self, k, d_common=64, n_iter_train=3,n_iter_test=5, d_slot=64, d_inputs=64, hid_dim=128):
        super(SlotAttention2, self).__init__()
        self.k = k
        self.d_common = d_common
        self.n_iter_train = n_iter_train
        self.n_iter_test = n_iter_test
        self.d_slot = d_slot
        self.d_inputs = d_inputs

        self.fc_q = nn.Linear(d_slot, d_common)
        self.fc_k = nn.Linear(d_inputs, d_common)
        self.fc_v = nn.Linear(d_inputs, d_common)

        self.gru = nn.GRUCell(d_common, d_slot)
        self.mlp = nn.Sequential(
            nn.Linear(d_slot, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, d_slot)
        )

        self.softmax = nn.Softmax(dim=2)
        self.mu = nn.Parameter(torch.randn(1, 1,d_common))
        self.sigma = nn.Parameter(torch.rand(1,1, d_common))

    
    def forward(self, inputs, name, out_dir):
        # inputs: (batch_size, n_inputs, d_inputs)
        # slots: (batch_size, n_slots, d_slot)
        if self.training:
            n_iter = self.n_iter_train
        else:
            n_iter = self.n_iter_train
        batch_size, n_inputs, d_inputs = inputs.size()
        mu = self.mu.expand(batch_size, self.k, -1).to(device)
        sigma = self.sigma.expand(batch_size, self.k, -1)
#         sigma = torch.ones(batch_size, self.k, self.d_common).to(device)*0.1
#         a = sigma < 0
#         if torch.any(a).item():
#         print("SIGMA: ", sigma)
        slots = torch.normal(mu, sigma).to(device)
        inputs = nn.LayerNorm(d_inputs).to(device)(inputs)
        k = self.fc_k(inputs)               # (batch_size, n_inputs, d_common)
        v = self.fc_v(inputs)               # (batch_size, n_inputs, d_common)
        masks = 0
        for i in range(n_iter):
            q = self.fc_q(nn.LayerNorm(self.d_slot).to(device)(slots))                # (batch_size, n_slots, d_common)
            attn = torch.bmm(k, q.permute(0, 2, 1)) / np.sqrt(self.d_common)            # (batch_size, n_inputs, n_slots)
            attn = self.softmax(attn) +  1e-8                                           # (batch_size, n_inputs, n_slots)
            attn = attn / attn.sum(dim=1, keepdim=True)                                 # (batch_size, n_inputs, n_slots)
            attn = attn.permute(0,2,1)
            updates = torch.einsum('bjd,bij->bid', v, attn)                             # (batch_size, n_slots, d_common)

            masks = attn
            slots = self.gru(updates.reshape(-1,self.d_common), slots.reshape(-1, self.d_slot)).reshape(batch_size, self.k, self.d_slot)
            slots = nn.LayerNorm(self.d_slot).to(device)(slots)
            slots = slots + self.mlp(slots)
        masks = masks.view(masks.shape[0], masks.shape[1], 128, 128)

        # masks = masks.squeeze(-1)
        for j in range(11):
            plt.imsave(out_dir + name + '_' + str(j) + '.png', masks[0, j].cpu().detach().numpy(), cmap='gray')
        
        return slots, masks
           

class RESBLOCK(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels, dropout=0.1, type_='nosample'):
        super(RESBLOCK, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.type = type_
        self.SiLU = nn.SiLU()
        if type_ == 'downsample':
            self.conv1 = nn.AvgPool2d(kernel_size=2, stride=2)
        elif type_ == 'upsample':
            # interpolate; this is an irrelevant line
            pass
#             self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.linear = nn.Linear(emb_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(out_channels,out_channels, kernel_size=3, padding=1, stride=1)
    
    def forward(self, x, time_embedding):
        x1 = self.norm1(x)
        x1 = self.SiLU(x1)
        if(self.type == 'upsample'):
            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners = False)
        else:
            x1 = self.conv1(x1)
        b, c, h, w = x1.size()
        # dimension of time_embedding = (batch_size, emb_channels)
        time_embedding = self.linear(self.SiLU(time_embedding))
        time_embedding = time_embedding.unsqueeze(-1).unsqueeze(-1)
        x1 = x1 + time_embedding
        x = self.norm2(x1)
        x = self.SiLU(x)
        x = self.dropout(x) # ??
        x = self.conv2(x)
        x = x + x1
        return x



class AttnBlock1(nn.Module):
    def __init__(self, in_channels, num_heads, slot_dim = 64):
        super(AttnBlock1, self).__init__()
        self.in_channels = in_channels
        self.self_attn = nn.MultiheadAttention(self.in_channels, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(self.in_channels)
        self.norm2 = nn.LayerNorm(self.in_channels)
        self.norm3 = nn.LayerNorm(self.in_channels)
        self.cross_attn = nn.MultiheadAttention(self.in_channels, num_heads, batch_first=True, kdim = slot_dim, vdim = slot_dim)
        self.linear1 = nn.Linear(self.in_channels, 4*self.in_channels)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(4*self.in_channels, self.in_channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, c):
        x = x + self.norm1(self.self_attn(x, x, x)[0])
        x = x + self.norm2(self.cross_attn(x, c, c)[0])
        x = x + self.norm3(self.linear2(self.dropout(self.gelu(self.linear1(x)))))
        return x
        


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, x):
        super(AttentionBlock, self).__init__()
        self.head_dim = 32
        self.channels = in_channels
        self.iter = x
        self.num_heads = in_channels // self.head_dim
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.onexone = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        # self.self_attn = []
        # self.cross_attn = []
        # self.ffn = []
        # for i in range(x):
        #     self.self_attn.append(nn.MultiheadAttention(self.channels, self.num_heads, batch_first=True))
        #     self.cross_attn.append(nn.MultiheadAttention(self.channels, self.num_heads, batch_first=True))
        #     self.ffn.append(nn.Sequential(
        #         nn.Linear(self.channels, self.channels),
        #         nn.SiLU(),
        #         nn.Linear(self.channels, self.channels)
        #     ))
        self.attn_blocks = nn.ModuleList([AttnBlock1(in_channels, self.num_heads) for i in range(x)])
        self.LN = nn.LayerNorm(self.channels)
        self.onexone2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        
    
    def forward(self, x, slots):
        x = self.norm1(x)
        x = self.onexone(x)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, h*w, c)
#         print(slots.shape)
        for block in self.attn_blocks:
            x = block(x, slots)
        x = x.reshape(b, h, w, c)
        x = x.permute(0, 3, 1, 2)
        x = self.onexone2(x)
        return x
            


class TimeEmbedding(nn.Module):
    def __init__(self, emb_channels):
        super(TimeEmbedding, self).__init__()
        self.emb_channels = emb_channels
        
    def forward(self, t):
        # dimension of t = (batch_size, 1)
        t = t.float()
        t = t.repeat(1, self.emb_channels)
        temp = torch.randn_like(t)
        for i in range(self.emb_channels):
            t[:, i] = t[:, i] / 10000.0**(2*i/self.emb_channels)
#             print("hahahhhahhaaaaa" ,i, t[:, i])
            if(i%2 == 0):
                temp[:, i] = torch.sin(t[:, i//2])
            else:
                temp[:, i] = torch.cos(t[:, i//2])
#                 print("ahhoaw",i//2,t[:, i//2])

        return temp
            



class UNET(nn.Module):
    def __init__(self, embedding_dim, img_size, transformer_iter, C=64):
        super(UNET, self).__init__()
        self.C = C
        self.embedding_dim = embedding_dim
        self.img_size = img_size
        self.transformer_iter = transformer_iter

        # ??
        self.conv_in = nn.Conv2d(in_channels=3, out_channels=C, kernel_size=3, padding=1, stride=1)
        self.R1 = RESBLOCK(C, C, embedding_dim)
        self.R2 = RESBLOCK(C, C, embedding_dim)
        self.D1 = RESBLOCK(C, C, embedding_dim, type_='downsample')
        self.R3 = RESBLOCK(C, 2*C, embedding_dim)
        self.T1 = AttentionBlock(2*C, transformer_iter)
        self.R4 = RESBLOCK(2*C, 2*C, embedding_dim)
        self.T2 = AttentionBlock(2*C, transformer_iter)
        self.D2 = RESBLOCK(2*C, 2*C, embedding_dim, type_='downsample')
        self.R5 = RESBLOCK(2*C, 3*C, embedding_dim)
        self.T3 = AttentionBlock(3*C, transformer_iter)
        self.R6 = RESBLOCK(3*C, 3*C, embedding_dim)
        self.T4 = AttentionBlock(3*C, transformer_iter)
        self.D3 = RESBLOCK(3*C, 3*C, embedding_dim, type_='downsample')
        self.R7 = RESBLOCK(3*C, 4*C, embedding_dim)
        self.T5 = AttentionBlock(4*C, transformer_iter)
        self.R8 = RESBLOCK(4*C, 4*C, embedding_dim)
        self.T6 = AttentionBlock(4*C, transformer_iter)

        self.R9 = RESBLOCK(4*C, 4*C, embedding_dim)
        self.T7 = AttentionBlock(4*C, transformer_iter)
        self.R10 = RESBLOCK(4*C, 4*C, embedding_dim)

        self.R11 = RESBLOCK(8*C, 4*C, embedding_dim)
        self.T8 = AttentionBlock(4*C, transformer_iter)
        self.R12 = RESBLOCK(8*C, 4*C, embedding_dim)
        self.T9 = AttentionBlock(4*C, transformer_iter)
        self.R13 = RESBLOCK(7*C, 4*C, embedding_dim)
        self.T10 = AttentionBlock(4*C, transformer_iter)
        self.U1 = RESBLOCK(4*C, 4*C, embedding_dim, type_='upsample')
        self.R14 = RESBLOCK(7*C, 3*C, embedding_dim)
        self.T11 = AttentionBlock(3*C, transformer_iter)
        self.R15 = RESBLOCK(6*C, 3*C, embedding_dim)
        self.T12 = AttentionBlock(3*C, transformer_iter)
        self.R16 = RESBLOCK(5*C, 3*C, embedding_dim)
        self.T13 = AttentionBlock(3*C, transformer_iter)
        self.U2 = RESBLOCK(3*C, 3*C, embedding_dim, type_='upsample')
        self.R17 = RESBLOCK(5*C, 2*C, embedding_dim)
        self.T14 = AttentionBlock(2*C, transformer_iter)
        self.R18 = RESBLOCK(4*C, 2*C, embedding_dim)
        self.T15 = AttentionBlock(2*C, transformer_iter)
        self.R19 = RESBLOCK(3*C, 2*C, embedding_dim)
        self.T16 = AttentionBlock(2*C, transformer_iter)
        self.U3 = RESBLOCK(2*C, 2*C, embedding_dim, type_='upsample')
        self.R20 = RESBLOCK(3*C, C, embedding_dim)
        self.R21 = RESBLOCK(2*C, C, embedding_dim)
        self.R22 = RESBLOCK(2*C, C, embedding_dim)
        self.norm = nn.GroupNorm(32, C)
        # ??
        self.conv_op = nn.Conv2d(in_channels=C, out_channels=3, kernel_size=3, padding=1, stride=1)

    def forward(self, x, time_embedding, slots):
        x = self.conv_in(x)
        conv_in = x
        x = self.R1(x, time_embedding)
        r1 = x
        x = self.R2(x, time_embedding)
        r2 = x
        x = self.D1(x, time_embedding)
        d1 = x
        x = self.R3(x, time_embedding)
        x = self.T1(x, slots)
        t1 = x
        x = self.R4(x, time_embedding)
        x = self.T2(x, slots)
        t2 = x
        x = self.D2(x, time_embedding)
        d2 = x
        x = self.R5(x, time_embedding)
        x = self.T3(x, slots)
        t3 = x
        x = self.R6(x, time_embedding)
        x = self.T4(x, slots)
        t4 = x
        x = self.D3(x, time_embedding)
        d3 = x
        x = self.R7(x, time_embedding)
        x = self.T5(x, slots)
        t5 = x
        x = self.R8(x, time_embedding)
        x = self.T6(x, slots)
        t6 = x

        x = self.R9(x, time_embedding)
        x = self.T7(x, slots)
        x = self.R10(x, time_embedding)

        x = torch.cat([x, t6], dim=1)
        x = self.R11(x, time_embedding)
        x = self.T8(x, slots)
        x = torch.cat([x, t5], dim=1)
        x = self.R12(x, time_embedding)
        x = self.T9(x, slots)
        x = torch.cat([x, d3], dim=1)
        x = self.R13(x, time_embedding)
        x = self.T10(x, slots)
        x = self.U1(x, time_embedding)
        x = torch.cat([x, t4], dim=1)
        x = self.R14(x, time_embedding)
        x = self.T11(x, slots)
        x = torch.cat([x, t3], dim=1)
        x = self.R15(x, time_embedding)
        x = self.T12(x, slots)
        x = torch.cat([x, d2], dim=1)
        x = self.R16(x, time_embedding)
        x = self.T13(x, slots)
        x = self.U2(x, time_embedding)
        x = torch.cat([x, t2], dim=1)
        x = self.R17(x, time_embedding)
        x = self.T14(x, slots)
        x = torch.cat([x, t1], dim=1)
        x = self.R18(x, time_embedding)
        x = self.T15(x, slots)
        x = torch.cat([x, d1], dim=1)
        x = self.R19(x, time_embedding)
        x = self.T16(x, slots)
        x = self.U3(x, time_embedding)
        x = torch.cat([x, r2], dim=1)
        x = self.R20(x, time_embedding)
        x = torch.cat([x, r1], dim=1)
        x = self.R21(x, time_embedding)
        x = torch.cat([x, conv_in], dim=1)
        x = self.R22(x, time_embedding)
        x = self.norm(x)
        x = self.conv_op(x)
        
        return x
        


class LDM(nn.Module):
    def __init__(self, vae):
        super(LDM, self).__init__()
        self.vae = vae
        
        self.unet = UNET(128, 32, 6)
        self.time_embedding_func = TimeEmbedding(128)
        self.encoder = CNNEncoder(64)
        self.sa = SlotAttention2(11, 64, 3, 5, 64, 64, 64)

        self.beta1 = 1e-4
        self.betaT = 2e-2
        self.alphas = torch.ones(1000)
        for i in range(1000):
            self.alphas[i] = self.alpha(i+1)
    def alpha(self, time_step):
        alphabar = 1.0
        for i in range(time_step):
            betat = self.beta1 + (self.betaT - self.beta1)*i/1000.0
            alphabar = alphabar * (1-betat)
        return alphabar
    def get_alphas(self, time):
        alphas = torch.randn_like(time.float())
        for i in range(time.size(0)):
            alphas[i][0] = self.alphas[time[i][0].item()-1]
        return alphas
    def get_xt_from_x0(self, x0 ,t, noise):
        alpha = self.get_alphas(t).unsqueeze(-1).unsqueeze(-1)
        mean = torch.sqrt(alpha) * x0
        std = torch.sqrt(1-alpha) * noise
        return mean + std
    def forward(self, x, noise):
        x_enc = self.encoder(x)
        slots,masks = self.sa(x_enc)
        with torch.no_grad():
            x = self.vae.encode(x)
        b = x.size(0)
        time_emb = torch.randint(1, 1001, (b, 1)).to(device)
#         print(".........",time_emb)
        xt = self.get_xt_from_x0(x, time_emb, noise)
        time_emb = self.time_embedding_func(time_emb)
#         print(time_emb)
#         print("time: ", time_emb)
        if np.isnan(xt[0][0][0][0].item()):
            print("NAN xt")
        x = self.unet(xt, time_emb, slots)
        if np.isnan(x[0][0][0][0].item()):
            print("NAN")
        return x
    def denoise(self, xprev, noise, t):
        alphas = self.alphas[t-1]
        beta = self.beta1 + (self.betaT - self.beta1)*t/1000.0
        z = torch.randn_like(xprev)
        if  t != 1:
            x = (1/np.sqrt((1-beta)))*(xprev - (beta*noise)/np.sqrt(1-alphas)) + np.sqrt(beta)*z
        else:
            x = (1/np.sqrt((1-beta)))*(xprev - (beta*noise)/np.sqrt(1-alphas)) 
        return x

    def infer(self, x, name, out_dir):
        x_enc = self.encoder(x)
        slots, masks = self.sa(x_enc, name, out_dir)
#         return masks
        with torch.no_grad():
            x = self.vae.encode(x)
        b = x.size(0)
        noise = torch.randn_like(x)
        xprev = noise
        for t in range(1000, 0, -1):
            time_emb = torch.tensor(t).to(device)
            time_emb = self.time_embedding_func(time_emb)
            x = self.unet(xprev, time_emb, slots)
            xprev = self.denoise(xprev, x, t)
        return self.vae.decode(xprev)


# %%
    
    

# %%
import argparse

def main():
    parser = argparse.ArgumentParser(description='Inference script for your model.')
    parser.add_argument('--input_dir', type=str, help='Path to the input directory', required=True)
    parser.add_argument('--model_path', type=str, help='Path to the trained model file', required=True)
    parser.add_argument('--part', type=int, choices=[1, 2], help='part', required=True)
    parser.add_argument('--output_dir', type=str, help='Path to the output directory', required=True)

    args = parser.parse_args()

    if(args.output_dir[-1]!='/'):
        args.output_dir += '/'
    if(args.input_dir[-1]!='/'):
        args.input_dir += '/'
    make_h5py('images.h5', args.input_dir)
    dataset = HDF5Dataset('images.h5')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if args.part == 1:
        model = SlotAttentionModel(11, 64, 3, 5, 64, 64, 64)
        model = nn.DataParallel(model)
        model.to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))

        model.eval()
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        filenames = sorted([i[:-4] for i in os.listdir(args.input_dir)])
        for i, inputs in enumerate(loader):
            # plt.imsave(args.output_dir + filenames[i] + '_orig.png', np.array(inputs[0].permute(1, 2, 0).cpu().numpy()))
            inputs = inputs.to(device)
            imgs, masks = model.forward(inputs, filenames[i], args.output_dir)
            masks = masks.squeeze(-1)
            for j in range(11):
                plt.imsave(args.output_dir + filenames[i] + '_' + str(j) + '.png', masks[0, j].cpu().detach().numpy(), cmap='gray')

    else:
        vae = VAE()
        ckpt = torch.load(vae_checkpoint_path)
        vae.load_state_dict(ckpt)
        SDM = LDM(vae)
        # SDM = nn.DataParallel(SDM)
        SDM.load_state_dict(torch.load(args.model_path, map_location=device))
        # SDM = nn.DataParallel(SDM)
        SDM.to(device)

        SDM.eval()
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        filenames = sorted([i[:-4] for i in os.listdir(args.input_dir)])
        for i, inputs in enumerate(loader):
            # plt.imsave(args.output_dir + filenames[i] + '_orig.png', np.array(inputs[0].permute(1, 2, 0).cpu().numpy()))
            inputs = inputs.to(device)
            outputs = SDM.infer(inputs, filenames[i], args.output_dir)
            plt.imsave(args.output_dir + filenames[i] + '.png' , outputs[0].permute(1, 2, 0).cpu().numpy())
        

if __name__ == "__main__":
    main()



# %%



