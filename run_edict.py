from edict_functions import EDICT_editing

im_path = "experiment_images/imagenet_cake.jpg"
base_prompt = "A cupcake"
target_prompt = "An Easter cupcake"


if __name__ == "__main__":
    EDICT_editing(im_path, base_prompt, target_prompt)
