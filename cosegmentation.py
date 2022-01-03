import argparse
import torch
from pathlib import Path
from torchvision import transforms
from extractor import ViTExtractor
from tqdm import tqdm
import numpy as np
import faiss
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple


def find_cosegmentation(image_paths: List[str], elbow: float = 0.975, load_size: int = 224, layer: int = 11,
                        facet: str = 'key', bin: bool = False, thresh: float = 0.065, model_type: str = 'dino_vits8',
                        stride: int = 4, votes_percentage: int = 75, sample_interval: int = 100,
                        remove_outliers: bool = False, outliers_thresh: float = 0.7, low_res_saliency_maps: bool = True,
                        save_dir: str = None) -> Tuple[List[Image.Image], List[Image.Image]]:
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
    :param remove_outliers: assume existence of outlier images and remove them from cosegmentation process.
    :param outliers_thresh: threshold on cosine similarity between cls descriptors to determine outliers.
    :param low_res_saliency_maps: Use saliency maps with lower resolution (dramatically reduces GPU RAM needs,
    doesn't deteriorate performance).
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
    if remove_outliers:
        cls_descriptors = []
    num_images = len(image_paths)
    if save_dir is not None:
        save_dir = Path(save_dir)

    # extract descriptors and saliency maps for each image
    for image_path in image_paths:
        image_batch, image_pil = extractor.preprocess(image_path, load_size)
        image_pil_list.append(image_pil)
        include_cls = remove_outliers  # removing outlier images requires the cls descriptor.
        descs = extractor.extract_descriptors(image_batch.to(device), layer, facet, bin, include_cls).cpu().numpy()
        curr_num_patches, curr_load_size = extractor.num_patches, extractor.load_size
        num_patches_list.append(curr_num_patches)
        load_size_list.append(curr_load_size)
        if remove_outliers:
            cls_descriptor, descs = torch.from_numpy(descs[:, :, 0, :]), descs[:, :, 1:, :]
            cls_descriptors.append(cls_descriptor)
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

        # save saliency maps and resized images if needed
        if save_dir is not None:
            fig, ax = plt.subplots()
            ax.axis('off')
            ax.imshow(saliency_maps_list[-1].reshape(num_patches_list[-1]), vmin=0, vmax=1, cmap='jet')
            fig.savefig(save_dir / f'{Path(image_path).stem}_saliency_map.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            image_pil.save(save_dir / f'{Path(image_path).stem}_resized.png')

    if remove_outliers:
        all_cls_descriptors = torch.stack(cls_descriptors, dim=2)[0, 0]
        mean_cls_descriptor = torch.mean(all_cls_descriptors, dim=0)[None, ...]
        cos_sim = torch.nn.CosineSimilarity(dim=1)
        similarities_to_mean = cos_sim(all_cls_descriptors, mean_cls_descriptor)
        inliers_idx = torch.where(similarities_to_mean >= outliers_thresh)[0]
        inlier_image_paths, outlier_image_paths = [], []
        inlier_descriptors, outlier_descriptors = [], []
        inlier_saliency_maps, outlier_saliency_maps = [], []
        inlier_image_pil, outlier_image_pil = [], []
        inlier_num_patches, outlier_num_patches = [], []
        inlier_load_size, outlier_load_size = [], []
        for idx, (image_path, descriptor, saliency_map, pil_image, num_patches, load_size) in enumerate(zip(image_paths,
                descriptors_list, saliency_maps_list, image_pil_list, num_patches_list, load_size_list)):
            (inlier_image_paths if idx in inliers_idx else outlier_image_paths).append(image_path)
            (inlier_descriptors if idx in inliers_idx else outlier_descriptors).append(descriptor)
            (inlier_saliency_maps if idx in inliers_idx else outlier_saliency_maps).append(saliency_map)
            (inlier_image_pil if idx in inliers_idx else outlier_image_pil).append(pil_image)
            (inlier_num_patches if idx in inliers_idx else outlier_num_patches).append(num_patches)
            (inlier_load_size if idx in inliers_idx else outlier_load_size).append(load_size)
        image_paths = inlier_image_paths
        descriptors_list = inlier_descriptors
        saliency_maps_list = inlier_saliency_maps
        image_pil_list = inlier_image_pil
        num_patches_list = inlier_num_patches
        load_size_list = inlier_load_size
        num_images = len(inliers_idx)

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
    labels_per_image = np.split(labels, np.cumsum(num_descriptors_per_image))

    if save_dir is not None:
        cmap = 'jet' if num_labels > 10 else 'tab10'
        for image_path, num_patches, label_per_image in zip(image_paths, num_patches_list, labels_per_image):
            fig, ax = plt.subplots()
            ax.axis('off')
            ax.imshow(label_per_image.reshape(num_patches), vmin=0, vmax=num_labels-1, cmap=cmap)
            fig.savefig(save_dir / f'{Path(image_path).stem}_clustering.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    # use saliency maps to vote for salient clusters
    votes = np.zeros(num_labels)
    for image_labels, saliency_map in zip(labels_per_image, saliency_maps_list):
        for label in range(num_labels):
            label_saliency = saliency_map[image_labels[:, 0] == label].mean()
            if label_saliency > thresh:
                votes[label] += 1
    salient_labels = np.where(votes >= np.ceil(num_images * votes_percentage / 100))
    # create masks using the salient labels
    segmentation_masks = []
    for img, labels, num_patches, load_size in zip(image_pil_list, labels_per_image, num_patches_list, load_size_list):
        mask = np.isin(labels, salient_labels).reshape(num_patches)
        resized_mask = np.array(Image.fromarray(mask).resize((load_size[1], load_size[0]), resample=Image.LANCZOS))
        # apply grabcut on mask
        grabcut_kernel_size = (7, 7)
        kernel = np.ones(grabcut_kernel_size, np.uint8)
        forground_mask = cv2.erode(np.uint8(resized_mask), kernel)
        forground_mask = np.array(Image.fromarray(forground_mask).resize(img.size, Image.NEAREST))
        background_mask = cv2.erode(np.uint8(1 - resized_mask), kernel)
        background_mask = np.array(Image.fromarray(background_mask).resize(img.size, Image.NEAREST))
        full_mask = np.ones((load_size[0], load_size[1]), np.uint8) * cv2.GC_PR_FGD
        full_mask[background_mask == 1] = cv2.GC_BGD
        full_mask[forground_mask == 1] = cv2.GC_FGD
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(np.array(img), full_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        grabcut_mask = np.where((full_mask == 2) | (full_mask == 0), 0, 1).astype('uint8')
        grabcut_mask = Image.fromarray(np.array(grabcut_mask, dtype=bool))
        segmentation_masks.append(grabcut_mask)

    if remove_outliers:
        outlier_segmentation_masks = []
        for load_size in outlier_load_size:
            outlier_segmentation_masks.append(Image.fromarray(np.zeros(load_size, dtype=bool)))

        final_segmentation_masks, final_pil_images = [], []
        for idx in range(len(image_paths)):
            if idx in inliers_idx:
                final_segmentation_masks.append(segmentation_masks.pop(0))
                final_pil_images.append(image_pil_list.pop(0))
            else:
                final_segmentation_masks.append(outlier_segmentation_masks.pop(0))
                final_pil_images.append(outlier_image_pil.pop(0))
        segmentation_masks = final_segmentation_masks
        image_pil_list = final_pil_images

    return segmentation_masks, image_pil_list


def draw_cosegmentation(segmentation_masks: List[Image.Image], pil_images: List[Image.Image]) -> List[plt.Figure]:
    """
    Visualizes cosegmentation results on chessboard background.
    :param segmentation_masks: list of binary segmentation masks
    :param pil_images: list of corresponding images.
    :return: list of figures with fg segment of each image and chessboard bg.
    """
    figures = []
    for seg_mask, pil_image in zip(segmentation_masks, pil_images):
        # make bg transparent in image
        np_image = np.array(pil_image)
        np_mask = np.array(seg_mask)
        stacked_mask = np.dstack(3 * [seg_mask])
        masked_image = np.array(pil_image)
        masked_image[~stacked_mask] = 0
        masked_image_transparent = np.concatenate((masked_image, 255. * np_mask.astype(np.int32)[..., None]), axis=-1)

        # create chessboard bg
        chessboard_bg = np.zeros(np_image.shape[:2])
        chessboard_edge = 10
        chessboard_bg[[x // chessboard_edge % 2 == 0 for x in range(chessboard_bg.shape[0])], :] = 1
        chessboard_bg[:, [x // chessboard_edge % 2 == 1 for x in range(chessboard_bg.shape[1])]] = \
            1 - chessboard_bg[:, [x // chessboard_edge % 2 == 1 for x in range(chessboard_bg.shape[1])]]
        chessboard_bg[chessboard_bg == 0] = 0.75
        chessboard_bg = 255. * chessboard_bg

        # show
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(chessboard_bg, cmap='gray', vmin=0, vmax=255)
        ax.imshow(masked_image_transparent.astype(np.int32), vmin=0, vmax=255)
        figures.append(fig)
    return figures


def draw_cosegmentation_binary_masks(segmentation_masks) -> List[plt.Figure]:
    """
    Visualize cosegmentation results as binary masks
    :param segmentation_masks: list of binary segmentation masks
    :return: list of figures with fg segment of each image and chessboard bg.
    """
    figures = []
    for seg_mask in segmentation_masks:
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(seg_mask)
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
    parser.add_argument('--remove_outliers', default='False', type=str2bool, help="Remove outliers using cls token.")
    parser.add_argument('--outliers_thresh', default=0.7, type=float, help="Threshold for removing outliers.")
    parser.add_argument('--low_res_saliency_maps', default='True', type=str2bool, help="using low resolution saliency "
                                                                                       "maps. Reduces RAM needs.")

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

            # computing cosegmentation
            seg_masks, pil_images = find_cosegmentation(curr_images, args.elbow, args.load_size, args.layer,
                                                        args.facet, args.bin, args.thresh, args.model_type, args.stride,
                                                        args.votes_percentage, args.sample_interval,
                                                        args.remove_outliers, args.outliers_thresh,
                                                        args.low_res_saliency_maps, curr_save_dir)

            # saving cosegmentations
            binary_mask_figs = draw_cosegmentation_binary_masks(seg_masks)
            chessboard_bg_figs = draw_cosegmentation(seg_masks, pil_images)

            for image, binary_fig, chessboard_fig in zip(curr_images, binary_mask_figs, chessboard_bg_figs):
                binary_fig.savefig(curr_save_dir / f'{Path(image).stem}_mask.png', bbox_inches='tight', pad_inches=0)
                chessboard_fig.savefig(curr_save_dir / f'{Path(image).stem}_vis.png', bbox_inches='tight', pad_inches=0)
            plt.close('all')
