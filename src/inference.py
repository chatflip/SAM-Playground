import argparse
import os
import random

import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the automatic mask generator"
    )
    parser.add_argument("--image_path", type=str, default="truck.jpg")
    args = parser.parse_args()
    return args


def visualize_mask(image, anns):
    height, width = image.shape[:2]
    if len(anns) == 0:
        return np.zeros((height, width, 4), dtype=np.uint8)
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    visualize_image = np.ones((height, width, 4))
    visualize_image[:, :, 3] = 0
    random.seed(0)
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        visualize_image[m] = color_mask
    visualize_image = (255.0 * visualize_image).astype(np.uint8)
    return visualize_image


def visualize_bbox(image, anns):
    visualize_image = image.copy()
    if len(anns) == 0:
        return visualize_image
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    random.seed(0)
    for ann in sorted_anns:
        bbox = ann["bbox"]
        color = np.random.random(3) * 255
        x, y, w, h = bbox
        visualize_image = cv2.rectangle(
            visualize_image,
            (x, y),
            (x + w, y + h),
            color,
            thickness=2,
        )
    return visualize_image


def load_model(model_type: str, checkpoint: str) -> SamAutomaticMaskGenerator:
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )
    return mask_generator


def main(args: argparse.Namespace) -> None:
    model_type = "vit_h"
    sam_checkpoint = os.path.join("weights", "sam_vit_h_4b8939.pth")
    mask_generator = load_model(model_type, sam_checkpoint)

    filename = os.path.basename(args.image_path)
    name, _ = os.path.splitext(filename)
    image_path = os.path.join(args.image_path)
    visualize_prefix = os.path.join("visualize", name)

    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    print(masks[0].keys())
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    visualized_bbox = visualize_bbox(image, masks)
    cv2.imwrite(f"{visualize_prefix}_bbox.png", visualized_bbox)

    visualized_mask = visualize_mask(image, masks)
    cv2.imwrite(f"{visualize_prefix}_mask.png", visualized_mask)


if __name__ == "__main__":
    args = config()
    main(args)
