import cv2
import numpy as np
import torch


def preprocess_image(image):
    # resize
    image = cv2.resize(image, (256, 256))

    # BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # normalize to [0,1]
    image = image / 255.0

    # SAME normalization as training (VERY IMPORTANT)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = (image - mean) / std

    # HWC → CHW
    image = np.transpose(image, (2, 0, 1))

    # to tensor + batch
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

    return image


def postprocess_mask(mask):
    # logits → probability
    mask = torch.sigmoid(mask)

    # threshold
    mask = (mask > 0.35).float()

    # remove batch + channel
    mask = mask.squeeze().cpu().numpy()

    return mask


def dice_score(pred, target):
    pred = torch.tensor(pred)
    target = torch.tensor(target)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    return (2 * intersection) / (union + 1e-6)


def iou_score(pred, target):
    pred = torch.tensor(pred)
    target = torch.tensor(target)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return intersection / (union + 1e-6)