import argparse
import torch
from extractor import ViTExtractor
import numpy as np
import matplotlib.pyplot as plt


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


def show_similarity_interactive(image_path_a: str, image_path_b: str, load_size: int = 224, layer: int = 11,
                                facet: str = 'key', bin: bool = False, stride: int = 4, model_type: str = 'dino_vits8',
                                num_sim_patches: int = 1):
    """
     finding similarity between a descriptor in one image to the all descriptors in the other image.
     :param image_path_a: path to first image.
     :param image_path_b: path to second image.
     :param load_size: size of the smaller edge of loaded images. If None, does not resize.
     :param layer: layer to extract descriptors from.
     :param facet: facet to extract descriptors from.
     :param bin: if True use a log-binning descriptor.
     :param stride: stride of the model.
     :param model_type: type of model to extract descriptors from.
     :param num_sim_patches: number of most similar patches from image_b to plot.
    """
    # extract descriptors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size
    image_batch_a, image_pil_a = extractor.preprocess(image_path_a, load_size)
    image_batch_b, image_pil_b = extractor.preprocess(image_path_b, load_size)
    descs_a = extractor.extract_descriptors(image_batch_a.to(device), layer, facet, bin, include_cls=True)
    num_patches_a, load_size_a = extractor.num_patches, extractor.load_size
    descs_b = extractor.extract_descriptors(image_batch_b.to(device), layer, facet, bin, include_cls=True)
    num_patches_b, load_size_b = extractor.num_patches, extractor.load_size

    # plot
    fig, axes = plt.subplots(1, 3)
    [axi.set_axis_off() for axi in axes.ravel()]
    visible_patches = []
    radius = patch_size // 2
    # plot image_a and the chosen patch. if nothing marked chosen patch is cls patch.
    axes[0].imshow(image_pil_a)

    # calculate and plot similarity between image1 and image2 descriptors
    similarities = chunk_cosine_sim(descs_a, descs_b)
    curr_similarities = similarities[0, 0, 0, 1:]  # similarity to all spatial descriptors, without cls token
    curr_similarities = curr_similarities.reshape(num_patches_b)
    axes[1].imshow(curr_similarities.cpu().numpy(), cmap='jet')

    # plot image_b and the closest patch in it to the chosen patch in image_a
    axes[2].imshow(image_pil_b)
    sims, idxs = torch.topk(curr_similarities.flatten(), num_sim_patches)
    for idx, sim in zip(idxs, sims):
        y_descs_coor, x_descs_coor = idx // num_patches_b[1], idx % num_patches_b[1]
        center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                  (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
        patch = plt.Circle(center, radius, color=(1, 0, 0, 0.75))
        axes[2].add_patch(patch)
        visible_patches.append(patch)
    plt.draw()

    # start interactive loop
    # get input point from user
    fig.suptitle('Select a point on the left image. \n Right click to stop.', fontsize=16)
    plt.draw()
    pts = np.asarray(plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))
    while len(pts) == 1:
        y_coor, x_coor = int(pts[0, 1]), int(pts[0, 0])
        new_H = patch_size / stride * (load_size_a[0] // patch_size - 1) + 1
        new_W = patch_size / stride * (load_size_a[1] // patch_size - 1) + 1
        y_descs_coor = int(new_H / load_size_a[0] * y_coor)
        x_descs_coor = int(new_W / load_size_a[1] * x_coor)

        # reset previous marks
        for patch in visible_patches:
            patch.remove()
            visible_patches = []

        # draw chosen point
        center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                  (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
        patch = plt.Circle(center, radius, color=(1, 0, 0, 0.75))
        axes[0].add_patch(patch)
        visible_patches.append(patch)

        # get and draw current similarities
        raveled_desc_idx = num_patches_a[1] * y_descs_coor + x_descs_coor
        reveled_desc_idx_including_cls = raveled_desc_idx + 1
        curr_similarities = similarities[0, 0, reveled_desc_idx_including_cls, 1:]
        curr_similarities = curr_similarities.reshape(num_patches_b)
        axes[1].imshow(curr_similarities.cpu().numpy(), cmap='jet')

        # get and draw most similar points
        sims, idxs = torch.topk(curr_similarities.flatten(), num_sim_patches)
        for idx, sim in zip(idxs, sims):
            y_descs_coor, x_descs_coor = idx // num_patches_b[1], idx % num_patches_b[1]
            center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                      (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
            patch = plt.Circle(center, radius, color=(1, 0, 0, 0.75))
            axes[2].add_patch(patch)
            visible_patches.append(patch)
        plt.draw()

        # get input point from user
        fig.suptitle('Select a point on the left image', fontsize=16)
        plt.draw()
        pts = np.asarray(plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))


""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate similarity inspection between two images.')
    parser.add_argument('--image_a', type=str, required=True, help='Path to the first image')
    parser.add_argument('--image_b', type=str, required=True, help='Path to the second image.')
    parser.add_argument('--load_size', default=224, type=int, help='load size of the input image.')
    parser.add_argument('--stride', default=4, type=int, help="""stride of first convolution layer. 
                                                                    small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                              Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                              vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                       options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=11, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='False', type=str2bool, help="create a binned descriptor if True.")
    parser.add_argument('--num_sim_patches', default=1, type=int, help="number of closest patches to show.")

    args = parser.parse_args()

    with torch.no_grad():

        show_similarity_interactive(args.image_a, args.image_b, args.load_size, args.layer, args.facet, args.bin,
                                    args.stride, args.model_type, args.num_sim_patches)
