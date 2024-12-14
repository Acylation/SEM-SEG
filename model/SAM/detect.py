import os
import cv2
import matplotlib.pyplot as plt
import torch
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

SCRIPT = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.dirname(os.path.abspath(__file__))
sample_path = os.path.join(script_dir, "silica.bmp") # check why reading file got denied
model_path = os.path.join(script_dir, "sam_vit_l_finetune_32.pth")
img = cv2.imread(sample_path)

import torch.nn as nn
class ModelSimple(nn.Module):
    """
    Wrapper for the sam model to to fine-tune the model on a new dataset

    ...
    Attributes:
    -----------
    freeze_encoder (bool): freeze the encoder weights
    freeze_decoder (bool): freeze the decoder weights
    freeze_prompt_encoder (bool): freeze the prompt encoder weights
    transform (ResizeLongestSide): resize the images to the model input size

    Methods:
    --------
    setup(): load the model and freeze the weights
    forward(images, points): forward pass of the model, returns the masks and iou_predictions
    """

    def __init__(self, freeze_encoder=True, freeze_decoder=False, freeze_prompt_encoder=True):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.freeze_prompt_encoder = freeze_prompt_encoder
        # we need this to make the input image size compatible with the model
        self.transform = ResizeLongestSide(1024) #This is 1024, because sam was trained on 1024x1024 images

    def setup(self):
        # TODO your path here
        self.model = sam_model_registry['vit_l'](os.path.join(SCRIPT, "sam_vit_l_0b3195.pth"))
        
        # to speed up training time, we normally freeze the encoder and decoder
        if self.freeze_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.freeze_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False
        self.transfrom = ResizeLongestSide(self.model.image_encoder.img_size)
    def forward(self, images):
        _, _, H, W = images.shape # batch, channel, height, width
        image_embeddings = self.model.image_encoder(images) # shape: (1, 256, 64, 64)
        # get prompt embeddings without acutally any prompts (uninformative)
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )

        # get low resolution masks and iou predictions
        # mulitmask_output=False means that we only get one mask per image,
        # otherwise we would get three masks per image
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings, # sparse_embeddings shape: (1, 0, 256)
            dense_prompt_embeddings=dense_embeddings, # dense_embeddings shape: (1, 256, 256)
            multimask_output=False,
        )
        # postprocess the masks to get the final masks and resize them to the original image size
        masks = F.interpolate(
            low_res_masks, # shape: (1, 1, 256, 256)
            (H, W),
            mode="bilinear",
            align_corners=False,
        )
        # shape masks after interpolate: torch.Size([1, 1, 1024, 1024])
        return masks, iou_predictions

# use my custom SAM model wrapper class
model_trained= ModelSimple()
model_trained.setup()
model_trained.load_state_dict(torch.load(os.path.join(SCRIPT, "sam_vit_l_finetune_32.pth")), strict=False)

with torch.no_grad():
   model_trained.load_state_dict(torch.load('finetuned.pth', map_location=torch.device('cpu')))
   model_trained.eval()
   img_size = model_trained.model.image_encoder.img_size
   print(f'img_size: {img_size}')
   print(f'img_size: {model_trained.model.image_encoder.img_size}')
    # TODO your code here