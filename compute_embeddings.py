import os

import imageio.v3 as imageio
import numpy as np
from micro_sam.util import get_sam_model, precompute_image_embeddings


def compute_embeddings(image_path, model_type, out_path):
    img = imageio.imread(image_path)
    model = get_sam_model(model_type=model_type)
    embed = precompute_image_embeddings(model, img, ndim=2)["features"]

    embed.tofile(out_path)
    embed_loaded = np.fromfile(out_path, dtype="float32").reshape(embed.shape)

    assert np.allclose(embed, embed_loaded)


def main():
    files = ["cells.png", "nuclei.png", "mitos-em.png", "vesicles-cryo.png"]
    models = ["vit_b_lm", "vit_b_lm", "vit_b_em_organelles", "vit_b_em_organelles"]
    for image, model in zip(files, models):
        name = image[:-4]
        out = f"embeddings_{name}.bin"
        if os.path.exists(out):
            continue
        compute_embeddings(image, model, out)


if __name__ == "__main__":
    main()
