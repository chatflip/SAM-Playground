import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def main() -> None:
    sam = sam_model_registry["vit_h"](checkpoint="weights/sam_vit_h_4b8939.pth")
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(model=sam)

    image = cv2.imread("images/truck.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

    print(masks)


if __name__ == "__main__":
    main()
