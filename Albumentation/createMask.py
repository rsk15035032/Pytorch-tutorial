import cv2
import numpy as np
from PIL import Image


def load_image(path):
    image = Image.open(path).convert("RGB")
    return np.array(image)



def create_empty_mask(shape):
    return np.zeros(shape, dtype=np.uint8)



def generate_mask(image):
    h, w, _ = image.shape
    mask = create_empty_mask((h, w, 3))

    # Convert to HSV for better color separation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # --------------------
    # Face / Skin (BLUE)
    # --------------------
    skin_lower = np.array([0, 20, 70])
    skin_upper = np.array([20, 255, 255])
    skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
    mask[skin_mask > 0] = (0, 0, 255)

    # --------------------
    # Eyes (RED)
    # --------------------
    eye_lower = np.array([0, 0, 0])
    eye_upper = np.array([180, 255, 40])
    eye_mask = cv2.inRange(hsv, eye_lower, eye_upper)
    mask[eye_mask > 0] = (255, 0, 0)

    # --------------------
    # Mouth / Lips (YELLOW)
    # --------------------
    lip_lower = np.array([160, 50, 50])
    lip_upper = np.array([180, 255, 255])
    lip_mask = cv2.inRange(hsv, lip_lower, lip_upper)
    mask[lip_mask > 0] = (255, 255, 0)

    # --------------------
    # Background (CYAN)
    # --------------------
    background = np.all(mask == [0, 0, 0], axis=-1)
    mask[background] = (0, 255, 255)

    return mask


def generate_appearance_mask(image):
    h, w, _ = image.shape
    mask = create_empty_mask((h, w, 3))

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Hair → BLACK
    hair_lower = np.array([0, 0, 0])
    hair_upper = np.array([180, 255, 50])
    hair = cv2.inRange(hsv, hair_lower, hair_upper)
    mask[hair > 0] = (0, 0, 0)

    # Face / neck → GREEN
    skin_lower = np.array([0, 20, 70])
    skin_upper = np.array([20, 255, 255])
    skin = cv2.inRange(hsv, skin_lower, skin_upper)
    mask[skin > 0] = (0, 255, 0)

    # Shirt → WHITE (bright low-sat)
    shirt_lower = np.array([0, 0, 200])
    shirt_upper = np.array([180, 40, 255])
    shirt = cv2.inRange(hsv, shirt_lower, shirt_upper)
    mask[shirt > 0] = (255, 255, 255)

    # Jacket / coat → RED
    coat_lower = np.array([0, 120, 70])
    coat_upper = np.array([10, 255, 255])
    coat = cv2.inRange(hsv, coat_lower, coat_upper)
    mask[coat > 0] = (255, 0, 0)

    # Background → BLUE
    bg = np.all(mask == [0, 0, 0], axis=-1)
    mask[bg] = (0, 0, 255)

    return mask


def save_mask(mask, path):
    Image.fromarray(mask).save(path)



if __name__ == "__main__":
    image_path = "images/shahrukh.jpg"
    output_mask_path = "images/mask.jpg"
    second_output_mask_path = "images/second_mask.jpg"

    image = load_image(image_path)

    mask = generate_mask(image)
    second_mask = generate_appearance_mask(image)

    save_mask(mask, output_mask_path)
    save_mask(second_mask, second_output_mask_path)

    print("✅ Semantic mask generated successfully")
