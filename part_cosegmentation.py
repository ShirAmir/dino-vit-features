import argparse
import torch
from pathlib import Path

import torchvision.transforms
from torchvision import transforms
from extractor import ViTExtractor
from tqdm import tqdm
import numpy as np
import faiss
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple
import pydensecrf.densecrf as dcrf
from matplotlib.colors import ListedColormap


def find_part_cosegmentation(image_paths: List[str], elbow: float = 0.975, load_size: int = 224, layer: int = 11,
                             facet: str = 'key', bin: bool = False, thresh: float = 0.065,
                             model_type: str = 'dino_vits8', stride: int = 4, votes_percentage: int = 75,
                             sample_interval: int = 100, low_res_saliency_maps: bool = True, num_parts: int = 4,
                             num_crop_augmentations: int = 0, three_stages: bool = False,
                             elbow_second_stage: float = 0.94, save_dir: str = None) -> Tuple[List[Image.Image],
                                                                                              List[Image.Image]]:
    """
    finding cosegmentation of a set of images.
    :param image_paths: a list of paths of all the images.
    :param elbow: elbow coefficient to set number of clusters.
    :param load_size: size of the smaller edge of loaded images. If None, does not resize.
    :param layer: layer to extract descriptors from.
    :param facet: facet to extract descriptors from.
    :param bin: if True use a log-binning descriptor.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    :param model_type: type of model to extract descriptors from.
    :param stride: stride of the model.
    :param votes_percentage: the percentage of positive votes so a cluster will be considered salient.
    :param sample_interval: sample every ith descriptor before applying clustering.
    :param low_res_saliency_maps: Use saliency maps with lower resolution (dramatically reduces GPU RAM needs,
    doesn't deteriorate performance).
    :param num_parts: Number of parts of final output.
    :param num_crop_augmentations: number of crop augmentations to apply on input images. Increases performance for
    small sets with high variations.
    :param three_stages: If true, uses three clustering stages - fg/bg, non-common objects, and parts. Increases
    performance for small sets with high variations.
    :param elbow_second_stage: elbow coefficient for clustering in the second stage.
    :param save_dir: optional. if not None save intermediate results in this directory.
    :return: a list of segmentation masks and a list of processed pil images.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    descriptors_list = []
    saliency_maps_list = []
    image_pil_list = []
    num_patches_list = []
    load_size_list = []
    if low_res_saliency_maps:
        saliency_extractor = ViTExtractor(model_type, stride=8, device=device)
    else:
        saliency_extractor = extractor
    num_images = len(image_paths)
    if save_dir is not None:
        save_dir = Path(save_dir)

    # create augmentations if needed
    if num_crop_augmentations > 0:
        augmentations_image_paths = []
        augmentations_dir = save_dir / 'augs'
        augmentations_dir.mkdir(exist_ok=True, parents=True)
        for image_path in image_paths:
            image_batch, image_pil = extractor.preprocess(image_path, load_size)
            flipped_image_pil = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
            crop_size = (int(image_batch.shape[2] * 0.95), int(image_batch.shape[3] * 0.95))
            random_crop = torchvision.transforms.RandomCrop(size=crop_size)
            for i in range(num_crop_augmentations):
                random_crop_file_name = augmentations_dir / f'{Path(image_path).stem}_resized_aug_{i}.png'
                random_crop(image_pil).save(random_crop_file_name)
                random_crop_flipped_file_name = augmentations_dir / f'{Path(image_path).stem}_resized_flip_aug_{i}.png'
                random_crop(flipped_image_pil).save(random_crop_flipped_file_name)
                augmentations_image_paths.append(random_crop_file_name)
                augmentations_image_paths.append(random_crop_flipped_file_name)
        image_paths = image_paths + augmentations_image_paths

    # extract descriptors and saliency maps for each image
    for image_path in image_paths:
        image_batch, image_pil = extractor.preprocess(image_path, load_size)
        image_pil_list.append(image_pil)
        descs = extractor.extract_descriptors(image_batch.to(device), layer, facet, bin).cpu().numpy()
        curr_num_patches, curr_load_size = extractor.num_patches, extractor.load_size
        num_patches_list.append(curr_num_patches)
        load_size_list.append(curr_load_size)
        descriptors_list.append(descs)
        if low_res_saliency_maps:
            if load_size is not None:
                low_res_load_size = (curr_load_size[0] // 2, curr_load_size[1] // 2)
            else:
                low_res_load_size = curr_load_size
            image_batch, image_pil = saliency_extractor.preprocess(image_path, low_res_load_size)

        saliency_map = saliency_extractor.extract_saliency_maps(image_batch.to(device)).cpu().numpy()
        curr_sal_num_patches, curr_sal_load_size = saliency_extractor.num_patches, saliency_extractor.load_size
        if low_res_saliency_maps:
            reshape_op = transforms.Resize(curr_num_patches, transforms.InterpolationMode.NEAREST)
            saliency_map = np.array(reshape_op(Image.fromarray(saliency_map.reshape(curr_sal_num_patches)))).flatten()
        saliency_maps_list.append(saliency_map)

        # save saliency maps and resized images if needed (not for augmentations)
        if save_dir is not None and not ('_aug_' in image_path.stem):
            fig, ax = plt.subplots()
            ax.axis('off')
            ax.imshow(saliency_maps_list[-1].reshape(num_patches_list[-1]), vmin=0, vmax=1, cmap='jet')
            fig.savefig(save_dir / f'{Path(image_path).stem}_saliency_map.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            image_pil.save(save_dir / f'{Path(image_path).stem}_resized.png')

    # cluster all images using k-means:
    all_descriptors = np.ascontiguousarray(np.concatenate(descriptors_list, axis=2)[0, 0])
    normalized_all_descriptors = all_descriptors.astype(np.float32)
    faiss.normalize_L2(normalized_all_descriptors)  # in-place operation
    sampled_descriptors_list = [x[:, :, ::sample_interval, :] for x in descriptors_list]
    all_sampled_descriptors = np.ascontiguousarray(np.concatenate(sampled_descriptors_list, axis=2)[0, 0])
    normalized_all_sampled_descriptors = all_sampled_descriptors.astype(np.float32)
    faiss.normalize_L2(normalized_all_sampled_descriptors)  # in-place operation

    sum_of_squared_dists = []
    n_cluster_range = list(range(1, 15))
    for n_clusters in n_cluster_range:
        algorithm = faiss.Kmeans(d=normalized_all_sampled_descriptors.shape[1], k=n_clusters, niter=300, nredo=10)
        algorithm.train(normalized_all_sampled_descriptors.astype(np.float32))
        squared_distances, labels = algorithm.index.search(normalized_all_descriptors.astype(np.float32), 1)
        objective = squared_distances.sum()
        sum_of_squared_dists.append(objective / normalized_all_descriptors.shape[0])
        if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow * sum_of_squared_dists[-2]):
            break

    num_labels = np.max(n_clusters) + 1
    num_descriptors_per_image = [num_patches[0]*num_patches[1] for num_patches in num_patches_list]
    labels_per_image = np.split(labels, np.cumsum(num_descriptors_per_image)[:-1])

    if save_dir is not None:
        cmap = 'jet' if num_labels > 10 else 'tab10'
        for image_path, num_patches, label_per_image in zip(image_paths, num_patches_list, labels_per_image):
            if not ('_aug_' in image_path.stem):
                fig, ax = plt.subplots()
                ax.axis('off')
                ax.imshow(label_per_image.reshape(num_patches), vmin=0, vmax=num_labels-1, cmap=cmap)
                fig.savefig(save_dir / f'{Path(image_path).stem}_clustering.png', bbox_inches='tight', pad_inches=0)
                plt.close(fig)

    # use saliency maps to vote for salient clusters (only original images vote, not augmentations)
    votes = np.zeros(num_labels)
    for image_path, image_labels, saliency_map in zip(image_paths, labels_per_image, saliency_maps_list):
        if not ('_aug_' in image_path.stem):
            for label in range(num_labels):
                label_saliency = saliency_map[image_labels[:, 0] == label].mean()
                if label_saliency > thresh:
                    votes[label] += 1
    salient_labels = np.where(votes >= np.ceil(num_images * votes_percentage / 100))[0]

    # cluster all parts using k-means:
    fg_masks = [np.isin(labels, salient_labels) for labels in labels_per_image]  # get only foreground descriptors
    fg_descriptor_list = [desc[:, :, fg_mask[:, 0], :] for fg_mask, desc in zip(fg_masks, descriptors_list)]
    all_fg_descriptors = np.ascontiguousarray(np.concatenate(fg_descriptor_list, axis=2)[0, 0])
    normalized_all_fg_descriptors = all_fg_descriptors.astype(np.float32)
    faiss.normalize_L2(normalized_all_fg_descriptors)  # in-place operation
    sampled_fg_descriptors_list = [x[:, :, ::sample_interval, :] for x in fg_descriptor_list]
    all_fg_sampled_descriptors = np.ascontiguousarray(np.concatenate(sampled_fg_descriptors_list, axis=2)[0, 0])
    normalized_all_fg_sampled_descriptors = all_fg_sampled_descriptors.astype(np.float32)
    faiss.normalize_L2(normalized_all_fg_sampled_descriptors)  # in-place operation

    sum_of_squared_dists = []
    # if applying three stages, use elbow to determine number of clusters in second stage, otherwise use the specified
    # number of parts.
    n_cluster_range = list(range(1, 15)) if three_stages else [num_parts]
    for n_clusters in n_cluster_range:
        part_algorithm = faiss.Kmeans(d=normalized_all_fg_sampled_descriptors.shape[1], k=n_clusters, niter=300, nredo=10)
        part_algorithm.train(normalized_all_fg_sampled_descriptors.astype(np.float32))
        squared_distances, part_labels = part_algorithm.index.search(normalized_all_fg_descriptors.astype(np.float32), 1)
        objective = squared_distances.sum()
        sum_of_squared_dists.append(objective / normalized_all_fg_descriptors.shape[0])
        if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow_second_stage * sum_of_squared_dists[-2]):
            break

    part_num_labels = np.max(part_labels) + 1
    parts_num_descriptors_per_image = [np.count_nonzero(mask) for mask in fg_masks]
    part_labels_per_image = np.split(part_labels, np.cumsum(parts_num_descriptors_per_image))

    # get smoothed parts using crf
    part_segmentations = []
    for img, num_patches, load_size, descs in zip(image_pil_list, num_patches_list, load_size_list, descriptors_list):
        bg_centroids = tuple(i for i in range(algorithm.centroids.shape[0]) if not i in salient_labels)
        curr_normalized_descs = descs[0, 0].astype(np.float32)
        faiss.normalize_L2(curr_normalized_descs)  # in-place operation
        # distance to parts
        dist_to_parts = ((curr_normalized_descs[:, None, :] - part_algorithm.centroids[None, ...]) ** 2
                         ).sum(axis=2)
        # dist to BG
        dist_to_bg = ((curr_normalized_descs[:, None, :] - algorithm.centroids[None, bg_centroids, :]) ** 2
                      ).sum(axis=2)
        min_dist_to_bg = np.min(dist_to_bg, axis=1)[:, None]
        d_to_cent = np.concatenate((dist_to_parts, min_dist_to_bg), axis=1).reshape(num_patches[0], num_patches[1],
                                                                                    part_num_labels + 1)
        d_to_cent = d_to_cent - np.max(d_to_cent, axis=-1)[..., None]
        upsample = torch.nn.Upsample(size=load_size)
        u = np.array(upsample(torch.from_numpy(d_to_cent).permute(2, 0, 1)[None, ...])[0].permute(1, 2, 0))
        d = dcrf.DenseCRF2D(u.shape[1], u.shape[0], u.shape[2])
        d.setUnaryEnergy(np.ascontiguousarray(u.reshape(-1, u.shape[-1]).T))
        compat = [50, 15]
        d.addPairwiseGaussian(sxy=(3, 3), compat=compat[0], kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)
        d.addPairwiseBilateral(sxy=5, srgb=13, rgbim=np.array(img), compat=compat[1], kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(10)
        final = np.argmax(Q, axis=0).reshape(load_size)
        parts_float = final.astype(np.float32)
        parts_float[parts_float == part_num_labels] = np.nan
        part_segmentations.append(parts_float)

    if three_stages:  # if needed, apply third stage

        # visualize second stage
        if num_crop_augmentations > 0:
            curr_part_segmentations, curr_image_pil_list = [], []
            for image_path, part_seg, pil_image in zip(image_paths, part_segmentations, image_pil_list):
                if not ('_aug_' in image_path.stem):
                    curr_part_segmentations.append(part_seg)
                    curr_image_pil_list.append(pil_image)
        else:
            curr_part_segmentations, curr_image_pil_list = part_segmentations, image_pil_list

        part_figs = draw_part_cosegmentation(part_num_labels, curr_part_segmentations, curr_image_pil_list)
        for image, part_fig in zip(curr_images, part_figs):
            part_fig.savefig(curr_save_dir / f'{Path(image).stem}_vis_sec_stage.png', bbox_inches='tight', pad_inches=0)
        plt.close('all')

        # get labels after crf for each descriptor
        smoothed_part_labels_per_image = []
        for part_segment, num_patches in zip(part_segmentations, num_patches_list):
            resized_part_segment = np.array(torch.nn.functional.interpolate(torch.from_numpy(part_segment)
                                                                            [None, None, ...], size=num_patches,
                                                                            mode='nearest')[0, 0])
            smoothed_part_labels_per_image.append(resized_part_segment.flatten())

        # take only parts that appear in all original images (otherwise they belong to non-common objects)
        votes = np.zeros(part_num_labels)
        for image_path, image_labels in zip(image_paths, smoothed_part_labels_per_image):
            if not ('_aug_' in image_path.stem):
                unique_labels = np.unique(image_labels[~np.isnan(image_labels)]).astype(np.int32)
                votes[unique_labels] += 1
        common_labels = np.where(votes == num_images)[0]

        # get labels after crf for each descriptor
        common_parts_masks = []
        for part_segment in smoothed_part_labels_per_image:
            common_parts_masks.append(np.isin(part_segment, common_labels).flatten())

        # cluster all final parts using k-means:
        common_descriptor_list = [desc[:, :, mask, :] for mask, desc in zip(common_parts_masks, descriptors_list)]
        all_common_descriptors = np.ascontiguousarray(np.concatenate(common_descriptor_list, axis=2)[0, 0])
        normalized_all_common_descriptors = all_common_descriptors.astype(np.float32)
        faiss.normalize_L2(normalized_all_common_descriptors)  # in-place operation
        sampled_common_descriptors_list = [x[:, :, ::sample_interval, :] for x in common_descriptor_list]
        all_common_sampled_descriptors = np.ascontiguousarray(np.concatenate(sampled_common_descriptors_list,
                                                                             axis=2)[0, 0])
        normalized_all_common_sampled_descriptors = all_common_sampled_descriptors.astype(np.float32)
        faiss.normalize_L2(normalized_all_common_sampled_descriptors)  # in-place operation

        common_part_algorithm = faiss.Kmeans(d=normalized_all_common_sampled_descriptors.shape[1], k=num_parts,
                                             niter=300, nredo=10)
        common_part_algorithm.train(normalized_all_common_sampled_descriptors.astype(np.float32))
        _, common_part_labels = part_algorithm.index.search(normalized_all_common_descriptors.astype(np.float32), 1)

        common_part_num_labels = np.max(common_part_labels) + 1
        parts_num_descriptors_per_image = [np.count_nonzero(mask) for mask in common_parts_masks]
        common_part_labels_per_image = np.split(common_part_labels, np.cumsum(parts_num_descriptors_per_image))

        # get smoothed parts using crf
        common_part_segmentations = []
        for img, num_patches, load_size, descs in zip(image_pil_list, num_patches_list, load_size_list, descriptors_list):
            bg_centroids_1 = tuple(i for i in range(algorithm.centroids.shape[0]) if not i in salient_labels)
            bg_centroids_2 = tuple(i for i in range(part_algorithm.centroids.shape[0]) if not i in common_labels)
            curr_normalized_descs = descs[0, 0].astype(np.float32)
            faiss.normalize_L2(curr_normalized_descs)  # in-place operation

            # distance to parts
            dist_to_parts = ((curr_normalized_descs[:, None, :] - common_part_algorithm.centroids[None, ...]) ** 2).sum(
                axis=2)
            # dist to BG
            dist_to_bg_1 = ((curr_normalized_descs[:, None, :] -
                             algorithm.centroids[None, bg_centroids_1, :]) ** 2).sum(axis=2)
            dist_to_bg_2 = ((curr_normalized_descs[:, None, :] -
                             part_algorithm.centroids[None, bg_centroids_2, :]) ** 2).sum(axis=2)
            dist_to_bg = np.concatenate((dist_to_bg_1, dist_to_bg_2), axis=1)
            min_dist_to_bg = np.min(dist_to_bg, axis=1)[:, None]
            d_to_cent = np.concatenate((dist_to_parts, min_dist_to_bg), axis=1).reshape(num_patches[0], num_patches[1],
                                                                                        num_parts + 1)
            d_to_cent = d_to_cent - np.max(d_to_cent, axis=-1)[..., None]
            upsample = torch.nn.Upsample(size=load_size)
            u = np.array(upsample(torch.from_numpy(d_to_cent).permute(2, 0, 1)[None, ...])[0].permute(1, 2, 0))
            d = dcrf.DenseCRF2D(u.shape[1], u.shape[0], u.shape[2])
            d.setUnaryEnergy(np.ascontiguousarray(u.reshape(-1, u.shape[-1]).T))

            compat = [50, 15]
            d.addPairwiseGaussian(sxy=(3, 3), compat=compat[0], kernel=dcrf.DIAG_KERNEL,
                                  normalization=dcrf.NORMALIZE_SYMMETRIC)
            d.addPairwiseBilateral(sxy=5, srgb=13, rgbim=np.array(img), compat=compat[1], kernel=dcrf.DIAG_KERNEL,
                                   normalization=dcrf.NORMALIZE_SYMMETRIC)
            Q = d.inference(10)
            final = np.argmax(Q, axis=0).reshape(load_size)
            common_parts_float = final.astype(np.float32)
            common_parts_float[common_parts_float == num_parts] = np.nan
            common_part_segmentations.append(common_parts_float)

        # reassign third stage results as final results
        part_segmentations = common_part_segmentations

    # remove augmentation results if existing
    if num_crop_augmentations > 0:
        no_aug_part_segmentations, no_aug_image_pil_list = [], []
        for image_path, part_seg, pil_image in zip(image_paths, part_segmentations, image_pil_list):
            if not ('_aug_' in image_path.stem):
                no_aug_part_segmentations.append(part_seg)
                no_aug_image_pil_list.append(pil_image)
        part_segmentations = no_aug_part_segmentations
        image_pil_list = no_aug_image_pil_list

    return part_segmentations, image_pil_list


def draw_part_cosegmentation(num_parts: int, segmentation_parts: List[np.ndarray], pil_images: List[Image.Image]) -> List[plt.Figure]:
    """
    Visualizes part cosegmentation results on chessboard background.
    :param num_parts: number of object parts in all part cosegmentations.
    :param segmentation_parts: list of binary segmentation masks
    :param pil_images: list of corresponding images.
    :return: list of figures with fg segment of each image and chessboard bg.
    """
    figures = []
    for parts_seg, pil_image in zip(segmentation_parts, pil_images):
        current_mask = ~np.isnan(parts_seg)  # np.isin(segmentation, segment_indexes)
        stacked_mask = np.dstack(3 * [current_mask])
        masked_image = np.array(pil_image)
        masked_image[~stacked_mask] = 0
        masked_image_transparent = np.concatenate((masked_image, 255. * current_mask.astype(np.uint8)[..., None]),
                                                  axis=-1)
        # create chessboard bg
        checkerboard_bg = np.zeros(masked_image.shape[:2])
        checkerboard_edge = 10
        checkerboard_bg[[x // checkerboard_edge % 2 == 0 for x in range(checkerboard_bg.shape[0])], :] = 1
        checkerboard_bg[:, [x // checkerboard_edge % 2 == 1 for x in range(checkerboard_bg.shape[1])]] = \
            1 - checkerboard_bg[:, [x // checkerboard_edge % 2 == 1 for x in range(checkerboard_bg.shape[1])]]
        checkerboard_bg[checkerboard_bg == 0] = 0.75
        checkerboard_bg = 255. * checkerboard_bg

        # show
        fig, ax = plt.subplots()
        ax.axis('off')
        color_list = ["red", "yellow", "blue", "lime", "darkviolet", "magenta", "cyan", "brown", "yellow"]
        cmap = 'jet' if num_parts > 10 else ListedColormap(color_list[:num_parts])
        ax.imshow(checkerboard_bg, cmap='gray', vmin=0, vmax=255)
        ax.imshow(masked_image_transparent.astype(np.int32), vmin=0, vmax=255)
        ax.imshow(parts_seg, cmap=cmap, vmin=0, vmax=num_parts - 1, alpha=0.5, interpolation='nearest')
        figures.append(fig)
    return figures


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
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor cosegmentations.')
    parser.add_argument('--root_dir', type=str, required=True, help='The root dir of image sets.')
    parser.add_argument('--save_dir', type=str, required=True, help='The root save dir for image sets results.')
    parser.add_argument('--load_size', default=None, type=int, help='load size of the input images. If None maintains'
                                                                    'original image size, if int resizes each image'
                                                                    'such that the smaller side is this number.')
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
    parser.add_argument('--thresh', default=0.065, type=float, help='saliency maps threshold to distinguish fg / bg.')
    parser.add_argument('--elbow', default=0.975, type=float, help='Elbow coefficient for setting number of clusters.')
    parser.add_argument('--votes_percentage', default=75, type=int, help="percentage of votes needed for a cluster to "
                                                                         "be considered salient.")
    parser.add_argument('--sample_interval', default=100, type=int, help="sample every ith descriptor for training"
                                                                         "clustering.")
    parser.add_argument('--outliers_thresh', default=0.7, type=float, help="Threshold for removing outliers.")
    parser.add_argument('--low_res_saliency_maps', default='True', type=str2bool, help="using low resolution saliency "
                                                                                       "maps. Reduces RAM needs.")
    parser.add_argument('--num_parts', default=4, type=int, help="Number of common object parts.")
    parser.add_argument('--num_crop_augmentations', default=0, type=int, help="If > 1, applies this number of random "
                                                                              "crop augmentations taking 95% of the "
                                                                              "original images and flip augmentations.")
    parser.add_argument('--three_stages', default=False, type=str2bool, help="If true, use three clustering stages "
                                                                             "instead of two. Useful for small sets "
                                                                             "with a lot of distraction objects.")
    parser.add_argument('--elbow_second_stage', default=0.94, type=float, help="Elbow coefficient for setting number "
                                                                               "of clusters.")

    args = parser.parse_args()

    with torch.no_grad():

        # prepare directories
        root_dir = Path(args.root_dir)
        sets_dir = [x for x in root_dir.iterdir() if x.is_dir()]
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        for set_dir in tqdm(sets_dir):
            print(f"working on {set_dir}")
            curr_images = [x for x in set_dir.iterdir() if x.suffix.lower() in ['.jpg', '.png', '.jpeg']]
            curr_save_dir = save_dir / set_dir.name
            curr_save_dir.mkdir(parents=True, exist_ok=True)

            # computing part cosegmentation
            parts_imgs, pil_images = find_part_cosegmentation(curr_images, args.elbow, args.load_size, args.layer,
                                                              args.facet, args.bin, args.thresh, args.model_type,
                                                              args.stride, args.votes_percentage, args.sample_interval,
                                                              args.low_res_saliency_maps, args.num_parts,
                                                              args.num_crop_augmentations, args.three_stages,
                                                              args.elbow_second_stage, curr_save_dir)

            # saving part cosegmentations
            part_figs = draw_part_cosegmentation(args.num_parts, parts_imgs, pil_images)

            for image, part_fig in zip(curr_images, part_figs):
                part_fig.savefig(curr_save_dir / f'{Path(image).stem}_vis.png', bbox_inches='tight', pad_inches=0)
            plt.close('all')
