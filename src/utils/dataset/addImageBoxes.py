import argparse, cv2, os, torch
import math as m
import numpy as np
import numpy.linalg as lg
import tifffile as tiff
import pandas as pd

from ultralytics import SAM
from tqdm import tqdm, trange

def draw_image_masks(image, bboxes, bgr_color=(255,128,0)):
    for bbox in bboxes:
        a_min, b_min, a_max, b_max = bbox
        cv2.rectangle(image, (a_min, b_min), (a_max, b_max), color=tuple(bgr_color), thickness=2)
    return(image)

# Apply SAM on an image with a /list/ of points. Functional, but doesn't provide good results (masks tend to overflow outside the forams)
def apply_sam_point_list(image, points:list, sam_model):
    labels = [1] * len(points) # label 1 means that point prompts are located inside the objects to segment (positive prompt)
    results = sam_model.predict(image, stream=False, points=points, labels=labels, imgsz = 1024)
    return results

# Apply SAM on an image with a /list/ of bounding boxes
def apply_sam_bbox_list(image, bboxes:list, sam_model):
    results = sam_model.predict(image, stream=False, bboxes=bboxes, imgsz = 1024)
    return results

def rectangle_intersection(bbox_1, bbox_2):
    if ((((bbox_2[0,0] <= bbox_1[0,0] <= bbox_2[1,0]) and (bbox_2[0,1] <= bbox_1[0,1] <= bbox_2[1,1])) or ((bbox_1[0,0] <= bbox_2[0,0] <= bbox_1[1,0]) and (bbox_1[0,1] <= bbox_2[0,1] <= bbox_1[1,1]))) or
        (((bbox_2[0,0] <= bbox_1[0,0] <= bbox_2[1,0]) and (bbox_2[0,1] <= bbox_1[1,1] <= bbox_2[1,1])) or ((bbox_1[0,0] <= bbox_2[0,0] <= bbox_1[1,0]) and (bbox_1[0,1] <= bbox_2[1,1] <= bbox_1[1,1]))) or
        (((bbox_2[0,0] <= bbox_1[1,0] <= bbox_2[1,0]) and (bbox_2[0,1] <= bbox_1[1,1] <= bbox_2[1,1])) or ((bbox_1[0,0] <= bbox_2[1,0] <= bbox_1[1,0]) and (bbox_1[0,1] <= bbox_2[1,1] <= bbox_1[1,1]))) or
        (((bbox_2[0,0] <= bbox_1[1,0] <= bbox_2[1,0]) and (bbox_2[0,1] <= bbox_1[0,1] <= bbox_2[1,1])) or ((bbox_1[0,0] <= bbox_2[1,0] <= bbox_1[1,0]) and (bbox_1[0,1] <= bbox_2[0,1] <= bbox_1[1,1])))):
        return(True)
    return(False)

def compute_max_point_frames2(point_coords, box_width, n_frames, n_mask_frames, tif_shape, mask_direction='z'):
    direction_dict = {'x':0, 'y':1, 'z':2}
    mask_axis = direction_dict[mask_direction.lower()]
    other_axis = np.delete(np.arange(3), mask_axis)
    half_box_width = box_width // 2
    n_min_frames = n_frames // 2 + 1
    n_max_frames = n_frames // 2 + n_mask_frames // 2
    n_points = point_coords.shape[0]
    direction_shape = [0, tif_shape[2-mask_axis]]
    min_mask_frame_array = np.zeros([n_points, 2], dtype='int32')
    min_mask_frame_array[:, 0] = point_coords[:, mask_axis] - n_max_frames
    min_mask_frame_array[:, 1] = point_coords[:, mask_axis] - n_min_frames
    max_mask_frame_array = np.zeros([n_points, 2], dtype='int32')
    max_mask_frame_array[:, 0] = point_coords[:, mask_axis] + n_min_frames
    max_mask_frame_array[:, 1] = point_coords[:, mask_axis] + n_max_frames
    label_frame_array = np.zeros([n_points, 2], dtype='int32')
    label_frame_array[:, 0] = point_coords[:, mask_axis] - n_frames // 2
    label_frame_array[:, 1] = point_coords[:, mask_axis] + n_frames // 2
    for i in range(n_points):
        for j in range(i+1, n_points):
            mask_interval_0 = np.array([[point_coords[i, mask_axis] - n_max_frames, point_coords[i, mask_axis] - n_min_frames],
                                        [point_coords[i, mask_axis] + n_min_frames, point_coords[i, mask_axis] + n_max_frames]])
            mask_interval_1 = np.array([[point_coords[j, mask_axis] - n_max_frames, point_coords[j, mask_axis] - n_min_frames],
                                        [point_coords[j, mask_axis] + n_min_frames, point_coords[j, mask_axis] + n_max_frames]])
            frame_interval_0 = np.array([point_coords[i, mask_axis] - n_frames // 2, point_coords[i, mask_axis] + n_frames // 2])
            frame_interval_1 = np.array([point_coords[j, mask_axis] - n_frames // 2, point_coords[j, mask_axis] + n_frames // 2])
            box_coords_0 = np.array([[point_coords[i, other_axis[0]] - half_box_width, point_coords[i, other_axis[1]] - half_box_width],
                                     [point_coords[i, other_axis[0]] + half_box_width, point_coords[i, other_axis[1]] + half_box_width]])
            box_coords_1 = np.array([[point_coords[j, other_axis[0]] - half_box_width, point_coords[j, other_axis[1]] - half_box_width],
                                     [point_coords[j, other_axis[0]] + half_box_width, point_coords[j, other_axis[1]] + half_box_width]])
            # If there is an intersection between the bounding boxes of the 2 points
            if rectangle_intersection(box_coords_0, box_coords_1):
                # If the first annotated frame of point 1 is inside the masked frame interval to the left of point 0
                if (mask_interval_0[0, 0] <= frame_interval_1[0] <= mask_interval_0[0, 1]):
                    min_mask_frame_array[i, :] = np.array([mask_interval_0[0,0], frame_interval_1[0] - 1], dtype='int32')
                # If the first annotated frame of point 0 is inside the masked frame interval to the left of point 1
                if (mask_interval_1[0, 0] <= frame_interval_0[0] <= mask_interval_1[0, 1]):
                    min_mask_frame_array[j, :] = np.array([mask_interval_1[0,0], frame_interval_0[0] - 1], dtype='int32')
                # If the first annotated frame of point 1 is inside the masked frame interval to the right of point 0
                if (mask_interval_0[1, 0] <= frame_interval_1[0] <= mask_interval_0[1, 1]):
                    max_mask_frame_array[i, :] = np.array([mask_interval_0[1,0], frame_interval_1[0] - 1], dtype='int32')
                    min_mask_frame_array[j, :] = np.array([mask_interval_0[1,0], frame_interval_1[0] - 1], dtype='int32')
                # If the first annotated frame of point 0 is inside the masked frame interval to the right of point 1
                if (mask_interval_1[1, 0] <= frame_interval_0[0] <= mask_interval_1[1, 1]):
                    max_mask_frame_array[j, :] = np.array([mask_interval_1[1,0], frame_interval_0[0] - 1], dtype='int32')
                    min_mask_frame_array[i, :] = np.array([mask_interval_1[1,0], frame_interval_0[0] - 1], dtype='int32')
                # If the last annotated frame of point 1 is inside the masked frame interval to the left of point 0
                if (mask_interval_0[0, 0] <= frame_interval_1[1] <= mask_interval_0[0, 1]):
                    min_mask_frame_array[i, :] = np.array([frame_interval_1[1] + 1, mask_interval_0[0,1]], dtype='int32')
                    max_mask_frame_array[j, :] = np.array([frame_interval_1[1] + 1, mask_interval_0[0,1]], dtype='int32')
                # If the last annotated frame of point 0 is inside the masked frame interval to the left of point 1
                if (mask_interval_1[0, 0] <= frame_interval_0[1] <= mask_interval_1[0, 1]):
                    min_mask_frame_array[j, :] = np.array([frame_interval_0[1] + 1, mask_interval_1[0,1]], dtype='int32')
                    max_mask_frame_array[i, :] = np.array([frame_interval_0[1] + 1, mask_interval_1[0,1]], dtype='int32')
                # If the last annotated frame of point 1 is inside the masked frame interval to the right of point 0
                if (mask_interval_0[1, 0] <= frame_interval_1[1] <= mask_interval_0[1, 1]):
                    max_mask_frame_array[i, :] = np.array([frame_interval_1[1] + 1, mask_interval_0[1,1]], dtype='int32')
                # If the last annotated frame of point 0 is inside the masked frame interval to the right of point 1
                if (mask_interval_1[1, 0] <= frame_interval_0[1] <= mask_interval_1[1, 1]):
                    max_mask_frame_array[j, :] = np.array([frame_interval_0[1] + 1, mask_interval_1[1,1]], dtype='int32')
    return(min_mask_frame_array, max_mask_frame_array, label_frame_array)


def compute_max_point_frames3(point_coords, box_width, n_frames, n_mask_frames, tif_shape, mask_direction='z'):
    direction_dict = {'x':0, 'y':1, 'z':2}
    mask_axis = direction_dict[mask_direction.lower()]
    other_axis = np.delete(np.arange(3), mask_axis)
    half_box_width = box_width // 2
    n_min_frames = n_frames // 2 + 1
    n_max_frames = n_frames // 2 + n_mask_frames // 2
    n_points = point_coords.shape[0]
    direction_shape = [0, tif_shape[2-mask_axis]]
    min_mask_frame_array = np.zeros([n_points, 2], dtype='int32')
    min_mask_frame_array[:, 0] = point_coords[:, mask_axis] - n_max_frames
    min_mask_frame_array[:, 1] = point_coords[:, mask_axis] - n_min_frames
    max_mask_frame_array = np.zeros([n_points, 2], dtype='int32')
    max_mask_frame_array[:, 0] = point_coords[:, mask_axis] + n_min_frames
    max_mask_frame_array[:, 1] = point_coords[:, mask_axis] + n_max_frames
    for i in range(n_points):
        for j in range(n_points):
            if (i != j):
                mask_interval_0 = np.array([[point_coords[i, mask_axis] - n_max_frames, point_coords[i, mask_axis] - n_min_frames],
                                            [point_coords[i, mask_axis] + n_min_frames, point_coords[i, mask_axis] + n_max_frames]])
                mask_interval_1 = np.array([[point_coords[j, mask_axis] - n_max_frames, point_coords[j, mask_axis] - n_min_frames],
                                            [point_coords[j, mask_axis] + n_min_frames, point_coords[j, mask_axis] + n_max_frames]])
                frame_interval_0 = np.array([point_coords[i, mask_axis] - n_frames // 2, point_coords[i, mask_axis] + n_frames // 2])
                frame_interval_1 = np.array([point_coords[j, mask_axis] - n_frames // 2, point_coords[j, mask_axis] + n_frames // 2])
                box_coords_0 = np.array([[point_coords[i, other_axis[0]] - half_box_width, point_coords[i, other_axis[1]] - half_box_width],
                                         [point_coords[i, other_axis[0]] + half_box_width, point_coords[i, other_axis[1]] + half_box_width]])
                box_coords_1 = np.array([[point_coords[j, other_axis[0]] - half_box_width, point_coords[j, other_axis[1]] - half_box_width],
                                         [point_coords[j, other_axis[0]] + half_box_width, point_coords[j, other_axis[1]] + half_box_width]])
                if rectangle_intersection(box_coords_0, box_coords_1):
                    if (mask_interval_0[0,0] <= frame_interval_1[0] <= mask_interval_0[0,1]):
                        min_mask_frame_array[i, :] = np.array([mask_interval_0[0,0], frame_interval_1[0] - 1], dtype='int32')
                    if (mask_interval_1[0,0] <= frame_interval_0[0] <= mask_interval_1[0,1]):
                        min_mask_frame_array[j, :] = np.array([mask_interval_1[0,0], frame_interval_0[0] - 1], dtype='int32')
                    if (mask_interval_0[1,0] <= frame_interval_1[0] <= mask_interval_0[1,1]):
                        max_mask_frame_array[i, :] = np.array([mask_interval_0[1,0], frame_interval_1[0] - 1], dtype='int32')
                        min_mask_frame_array[j, :] = np.array([mask_interval_0[1,0], frame_interval_1[0] - 1], dtype='int32')
                    if (mask_interval_1[1,0] <= frame_interval_0[0] <= mask_interval_1[1,1]):
                        max_mask_frame_array[j, :] = np.array([mask_interval_1[1,0], frame_interval_0[0] - 1], dtype='int32')
                        min_mask_frame_array[i, :] = np.array([mask_interval_1[1,0], frame_interval_0[0] - 1], dtype='int32')
                    if (mask_interval_0[0,0] <= frame_interval_1[1] <= mask_interval_0[0,1]):
                        min_mask_frame_array[i, :] = np.array([frame_interval_1[1] + 1, mask_interval_0[0,1]], dtype='int32')
                        max_mask_frame_array[j, :] = np.array([frame_interval_1[1] + 1, mask_interval_0[0,1]], dtype='int32')
                    if (mask_interval_1[0,0] <= frame_interval_0[1] <= mask_interval_1[0,1]):
                        min_mask_frame_array[j, :] = np.array([frame_interval_0[1] + 1, mask_interval_1[0,1]], dtype='int32')
                        max_mask_frame_array[i, :] = np.array([frame_interval_0[1] + 1, mask_interval_1[0,1]], dtype='int32')
                    if (mask_interval_0[1,0] <= frame_interval_1[1] <= mask_interval_0[1,1]):
                        max_mask_frame_array[i, :] = np.array([frame_interval_1[1] + 1, mask_interval_0[1,1]], dtype='int32')
                    if (mask_interval_1[1,0] <= frame_interval_0[1] <= mask_interval_1[1,1]):
                        max_mask_frame_array[j, :] = np.array([frame_interval_0[1] + 1, mask_interval_1[1,1]], dtype='int32')
    return(min_mask_frame_array, max_mask_frame_array)

def compute_max_point_frames(point_coords, box_width, n_frames, n_mask_frames, tif_shape, mask_direction='z'):
    direction_dict = {'x':0, 'y':1, 'z':2}
    mask_axis = direction_dict[mask_direction.lower()]
    other_axis = np.delete(np.arange(3), mask_axis)
    half_box_width = box_width // 2
    n_min_frames = n_frames // 2 + 1
    n_max_frames = n_frames // 2 + n_mask_frames // 2
    n_points = point_coords.shape[0]
    direction_shape = [0, tif_shape[2-mask_axis]]
    min_mask_frame_array = np.zeros([n_points, 2], dtype='int32')
    min_mask_frame_array[:, 0] = point_coords[:, mask_axis] - n_max_frames
    min_mask_frame_array[:, 1] = point_coords[:, mask_axis] - n_min_frames
    max_mask_frame_array = np.zeros([n_points, 2], dtype='int32')
    max_mask_frame_array[:, 0] = point_coords[:, mask_axis] + n_min_frames
    max_mask_frame_array[:, 1] = point_coords[:, mask_axis] + n_max_frames
    for i in range(n_points):
        for j in range(i+1, n_points):
            frame_interval_0 = np.array([point_coords[i, mask_axis] - n_frames // 2, point_coords[i, mask_axis] + n_frames // 2])
            frame_interval_1 = np.array([point_coords[j, mask_axis] - n_frames // 2, point_coords[j, mask_axis] + n_frames // 2])
            box_coords_0 = np.array([[point_coords[i, other_axis[0]] - half_box_width, point_coords[i, other_axis[1]] - half_box_width],
                                     [point_coords[i, other_axis[0]] + half_box_width, point_coords[i, other_axis[1]] + half_box_width]])
            box_coords_1 = np.array([[point_coords[j, other_axis[0]] - half_box_width, point_coords[j, other_axis[1]] - half_box_width],
                                     [point_coords[j, other_axis[0]] + half_box_width, point_coords[j, other_axis[1]] + half_box_width]])
            if rectangle_intersection(box_coords_0, box_coords_1):
                if (min_mask_frame_array[i, 0] <= frame_interval_1[0] <= min_mask_frame_array[i, 1]):
                    min_mask_frame_array[i, :] = np.array([max(min_mask_frame_array[i, 0], min_mask_frame_array[i, 0]), min(min_mask_frame_array[i, 1], frame_interval_1[0] - 1)], dtype='int32')
                if (min_mask_frame_array[j, 0] <= frame_interval_0[0] <= min_mask_frame_array[j, 1]):
                    min_mask_frame_array[j, :] = np.array([max(min_mask_frame_array[j, 0], min_mask_frame_array[j, 0]), min(min_mask_frame_array[j, 1], frame_interval_0[0] - 1)], dtype='int32')
                if (max_mask_frame_array[i, 0] <= frame_interval_1[0] <= max_mask_frame_array[i, 1]):
                    max_mask_frame_array[i, :] = np.array([max(max_mask_frame_array[i, 0], max_mask_frame_array[i, 0]), min(max_mask_frame_array[i, 1], frame_interval_1[0] - 1)], dtype='int32')
                    min_mask_frame_array[j, :] = np.array([max(min_mask_frame_array[j, 0], min_mask_frame_array[j, 0]), min(min_mask_frame_array[j, 1], frame_interval_1[0] - 1)], dtype='int32')
                if (max_mask_frame_array[j, 0] <= frame_interval_0[0] <= max_mask_frame_array[j, 1]):
                    max_mask_frame_array[j, :] = np.array([max(max_mask_frame_array[j, 0], max_mask_frame_array[j, 0]), min(max_mask_frame_array[j, 1], frame_interval_0[0] - 1)], dtype='int32')
                    min_mask_frame_array[i, :] = np.array([max(min_mask_frame_array[i, 0], min_mask_frame_array[i, 0]), min(min_mask_frame_array[i, 1], frame_interval_0[0] - 1)], dtype='int32')
                if (min_mask_frame_array[i, 0] <= frame_interval_1[1] <= min_mask_frame_array[i, 1]):
                    min_mask_frame_array[i, :] = np.array([max(min_mask_frame_array[i, 0], frame_interval_1[1] + 1), min(min_mask_frame_array[i, 1], min_mask_frame_array[i, 1])], dtype='int32')
                    max_mask_frame_array[j, :] = np.array([max(max_mask_frame_array[j, 0], frame_interval_1[1] + 1), min(max_mask_frame_array[j, 1], min_mask_frame_array[i, 1])], dtype='int32')
                if (min_mask_frame_array[j, 0] <= frame_interval_0[1] <= min_mask_frame_array[j, 1]):
                    min_mask_frame_array[j, :] = np.array([max(min_mask_frame_array[j, 0], frame_interval_0[1] + 1), min(min_mask_frame_array[j, 1], min_mask_frame_array[j, 1])], dtype='int32')
                    max_mask_frame_array[i, :] = np.array([max(max_mask_frame_array[i, 0], frame_interval_0[1] + 1), min(max_mask_frame_array[i, 1], min_mask_frame_array[j, 1])], dtype='int32')
                if (max_mask_frame_array[i, 0] <= frame_interval_1[1] <= max_mask_frame_array[i, 1]):
                    max_mask_frame_array[i, :] = np.array([max(max_mask_frame_array[i, 0], frame_interval_1[1] + 1), min(max_mask_frame_array[i, 1], max_mask_frame_array[i, 1])], dtype='int32')
                if (max_mask_frame_array[j, 0] <= frame_interval_0[1] <= max_mask_frame_array[j, 1]):
                    max_mask_frame_array[j, :] = np.array([max(max_mask_frame_array[j, 0], frame_interval_0[1] + 1), min(max_mask_frame_array[j, 1], max_mask_frame_array[j, 1])], dtype='int32')
    return(min_mask_frame_array, max_mask_frame_array)

def draw_bboxes(image, bboxes, color=(128,128,0), thickness=-1):
    for bbox in bboxes:
        a_min, b_min, a_max, b_max = bbox
        cv2.rectangle(image, (a_min, b_min), (a_max, b_max), color=tuple(color), thickness=thickness)
    return(image)

def bboxes_from_2D_point_annotations(point_annotations: list[list[int]], box_width: int, current_frame_shape: tuple[int,int]):
    """
    Generate bounding boxes from 2D point annotations.
    Args:
        point_annotations (list[list[int]]): 
            Array of 2D point annotations, where each annotation is a tuple (x, y).
        box_width (int): 
            The width (and height) of the square bounding box to generate around each point.
        current_frame_shape (tuple[int, int]): 
            The shape of the current frame as (height, width), used to clip bounding boxes within image boundaries.
    Returns:
        list[list[int, ...]]: 
            A list of bounding boxes in [xmin, ymin, xmax, ymax] format for each point annotation.
    """
    xyxy_annotations_bboxes: list[list[int]] = []
    for xyz_annotation in point_annotations:
        a_min = int(max(xyz_annotation[0] - box_width // 2, 0))
        a_max = int(min(xyz_annotation[0] + box_width // 2, current_frame_shape[0]))
        b_min = int(max(xyz_annotation[1] - box_width // 2, 0))
        b_max = int(min(xyz_annotation[1] + box_width // 2, current_frame_shape[1]))
        xyxy_annotations_bboxes.append([a_min, b_min, a_max, b_max]) # upper left corner, lower right corner
    return(xyxy_annotations_bboxes)

# New function that will replace save_annotations...
def generate_bboxes_and_masks(tif_path, csv_path, output_folder, box_width, nb_dup_annotations, 
                              axis_indices, model_path, base_image_name, apply_sam_bboxes, apply_sam_points, 
                              nb_masks, gray_value=128, csv_separator=';'):
    """Compute bounding boxes for YOLO training from a 3D image and a set of initial point annotations. 
    In the following, initial annotations are denoted (xyz)i ; YOLO annotations are denoted (xywhn)
    The 3D image is processed as a stack of 2D images, e.g. along the z axis.
    Bounding boxes can be arbitrarily defined with fixed dimensions, e.g. 40 pixels width-height
    Bounding boxes can also derived from a SAM assisted segmentation using (xyz)i as point prompts, or boxes surrounding (xyz)i as bounding boxes prompts.
    Initial point annotations (xyz)i can be duplicated above and below the slice z, i.e. z-1, z+1, z-2 etc. The number of duplications (above + below) is defined by num_frames.
    Once duplicated, the (xyz) annotations can be used directly as prompts for SAM segmentation or by the intermediate of bounding boxes.
    Masks can be drawn in the images above and below the duplicated annotations to prevent parts of the forams to be visible but not annotated.

    Args:
        tif_path (string): path to the 3D image file
        csv_path (string): path to the (x,y,z)i ground truth annotations
        output_folder (string): where to save the modified 2D images
        box_width (int): width of the bounding box centered on the ground truth annotations
        nb_dup_annotations (int): number of annotations duplicated before and after the ground truth annotation
        axis_indices (int): indices of the axes to prcess (0:x; 1:y; 2:z)
        model_path (string): path to the SAM model
        base_image_name (string): prefix for the output image
        apply_sam_bboxes (boolean): segment the foram based on the bounding box centered on the annotation
        apply_sam_points (boolean): segment the foram based on the point annotation
        nb_masks (int): number of masks that will be drawn before and after the duplicated annotations
        gray_value (int, optional): color of the masks. Defaults to 128.
        csv_separator (str, optional): Defaults to ';'.
    """
    direction_dict = {0:'x', 1:'y', 2:'z'}
    draw_mask = bool(nb_masks > 1)
    bgr_color = (gray_value, gray_value, gray_value)
    # TODO: check that all annotations are within the bounds of the image?
    
    if apply_sam_bboxes or apply_sam_points:
        sam_model = SAM(model_path)
        print("MODELE ", type(sam_model))
    
    # Create the output folder and necessary subfolders
    os.makedirs(output_folder, exist_ok=True)
    images_folder = os.path.join(output_folder, "images")
    labels_folder = os.path.join(output_folder, "labels")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    # Load CSV data
    df = pd.read_csv(csv_path, delimiter=csv_separator)
    xyz_annotations_init = df.to_numpy()[:, 1:]

    # Load the TIFF file, reorder its dimensions to make them coherent with the annotations: zyx => xyz
    tif_data = tiff.imread(tif_path) 
    tif_shape: tuple = tif_data.shape # plane, row, column
    print("Tif shape (z, y, x): ", tif_shape)
    img_shape:np.ndarray = [tif_shape[2], tif_shape[1], tif_shape[0]] # 0: x, 1: y, 2: z
    print("Image Dimensions [x, y, z]: ", img_shape)

    # Duplicate xyz annotations and compute masks centers, 1 axis at a time
    relative_indices_annotations = np.arange(-nb_dup_annotations//2, nb_dup_annotations//2 + 1) # Ex: -2, -1, 0, +1, +2
    relative_indices_masks_half_range = np.arange(nb_dup_annotations//2+1, (nb_dup_annotations+nb_masks)//2+1) # Ex.: 3, 4, 5
    relative_indices_masks = np.hstack((np.flip(-relative_indices_masks_half_range), relative_indices_masks_half_range)) # Ex.: -5, -4, -3, +3, +4, +5
    for axis in axis_indices: # 0: x, 1: y, 2: z
        d = direction_dict[axis]
        print(f"Current direction : axis {d.upper()}, {axis}")
        xyz_annotations_dup = []
        xyz_masks = []
        for xyz_annotation in xyz_annotations_init:
            # Annotations: Copy xyz annotation, then update coordinates. Ex.: z => z-2, z-1, z, z+1, z+2
            xyz_annotation_copied = np.tile(xyz_annotation, (nb_dup_annotations + 1, 1)) # +1 is meant for including the initial annotation. Ex.: z, z, z, z, z
            xyz_annotation_copied[:,axis] = xyz_annotation[axis] + relative_indices_annotations # Update coordinates. Ex.: z-2, z-1, z, z+1, z+2
            xyz_annotations_dup.append(xyz_annotation_copied)
            # Masks: Copy xyz annotation, then update coordinates. Ex.: z =>  z-5, z-4, z-3, z+3, z+4, z+5
            xyz_masks_current = np.tile(xyz_annotation, (nb_masks, 1)) # Ex.: z, z, z, z, z, z
            xyz_masks_current[:,axis] = xyz_annotation[axis] + relative_indices_masks # Ex.: z-5, z-4, z-3, z+3, z+4, z+5
            xyz_masks.append(xyz_masks_current)

        # Delete rows with out of bound values (i.e. outside 0 and the dimension along the current axis)
        xyz_annotations_dup = np.vstack(xyz_annotations_dup) # Convert list to nx3 array
        xyz_annotations_dup = xyz_annotations_dup[(xyz_annotations_dup[:, axis] >= 0) & (xyz_annotations_dup[:, axis] < img_shape[axis])]
        xyz_masks = np.vstack(xyz_masks)
        xyz_masks = xyz_masks[(xyz_masks[:, axis] >= 0) & (xyz_masks[:, axis] < img_shape[axis])]

        # Loop over the frames along the current axis
        for idx in trange((img_shape[axis]), desc="Processing images"):
            # Extract current frame. By convention we have 0=x, 1=y, 2=z ; but numpy has 0=z, 1=y, 2=x => permute 2 and 0 for selecting the frame
            frame = np.take(tif_data, indices=idx, axis=abs(axis-2)).copy()
            # SAM expects color image
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) #W, H, C

            # Get ground truth annotations
            xyz_current_annotations:np.ndarray = xyz_annotations_dup[xyz_annotations_dup[:,axis] == idx]
            if xyz_current_annotations.shape[0] == 0:
                continue
            xy_current_annotations:list[list[int]] = np.delete(xyz_current_annotations, axis, axis=1).tolist() # Remove column whose index corresponds to the current axis. E.g.: x1,y1,z ; x2,y2,z... => x1,y1 ; x2,y2...

            # Generate annotations bounding boxes
            xyxy_annotations_bboxes:list[list[int]] = bboxes_from_2D_point_annotations(xy_current_annotations, box_width, frame.shape)

            # Yolo format annotations (xywhn)
            # TODO: arrange the conditional tests
            xywhn_annotations = []
            if apply_sam_bboxes:
                sam_results:list = apply_sam_bbox_list(frame, xyxy_annotations_bboxes, sam_model)
            if apply_sam_points:
                sam_results:list = apply_sam_point_list(frame, xy_current_annotations, sam_model)
                #TODO : ignore predicted masks whose size exceeds the bounding box?
            if apply_sam_bboxes or apply_sam_points:
                for pred_box in sam_results[0].boxes: # Results is a list, with 1 element/image (i.e. 1 element in our case)
                    x_center, y_center, width, height = pred_box.xywhn[0] # YOLO format normalized bounding box is already stored in SAM results
                    xywhn_annotations.append([0, x_center, y_center, width, height]) #TOCHECK: what is the 0 for at line start?
            else:
                for bbox in xyxy_annotations_bboxes:
                    x_center = (bbox[0] + bbox[2]) / (2 * frame.shape[1])
                    y_center = (bbox[1] + bbox[3]) / (2 * frame.shape[0])
                    width = (bbox[2] - bbox[0]) / frame.shape[1]
                    height = (bbox[3] - bbox[1]) / frame.shape[0]
                    xywhn_annotations.append([0, x_center, y_center, width, height])

            # Copy patches of the image before drawing the masks
            temp_copy_bboxes_content = np.zeros(frame.shape)
            for xyxy_annotation_bbox in xyxy_annotations_bboxes:
                temp_copy_bboxes_content[xyxy_annotation_bbox[1]:xyxy_annotation_bbox[3]+1, xyxy_annotation_bbox[0]:xyxy_annotation_bbox[2]+1] = frame[xyxy_annotation_bbox[1]:xyxy_annotation_bbox[3]+1, xyxy_annotation_bbox[0]:xyxy_annotation_bbox[2]+1]

            # Draw masks!
            xyz_current_masks = xyz_masks[xyz_masks[:,axis] == idx] # Get masks centers for the current frame
            xyz_current_masks = np.delete(xyz_current_masks, axis, axis=1) # Remove column whose index corresponds to the current axis. E.g.: x1,y1,z ; x2,y2,z... => x1,y1 ; x2,y2...
            xyxy_masks = bboxes_from_2D_point_annotations(xyz_current_masks, box_width, frame.shape)
            frame = draw_bboxes(frame, xyxy_annotations_bboxes, thickness=-1)
            
            # paste patches located in the annotations boxes
            frame[temp_copy_bboxes_content[:] != 0] = temp_copy_bboxes_content[temp_copy_bboxes_content[:] != 0]

            # Save the image
            output_image_path = os.path.join(images_folder, f'{base_image_name}_{d}_{idx}.png')
            cv2.imwrite(output_image_path, frame)

            # Save the YOLO formatted bounding boxes

            # Save the SAM results (for statistics)

    exit()

    # Generate xywhn bounding boxes

# Soon to be deprecated
def save_images_and_annotations(tif_path, csv_path, output_folder, box_width, num_frames, axis_indices, model_path, base_image_name, apply_sam_bboxes, apply_sam_points, num_frames_to_mask, gray_value=128, csv_separator=';'):
    direction_dict = {0:'z', 1:'y', 2:'x'}
    draw_mask = bool(num_frames_to_mask > 1)
    bgr_color = (gray_value, gray_value, gray_value)
    
    if apply_sam_bboxes or apply_sam_points:
        sam_model = SAM(model_path)
    
    # Create the output folder and necessary subfolders
    os.makedirs(output_folder, exist_ok=True)
    images_folder = os.path.join(output_folder, "images")
    labels_folder = os.path.join(output_folder, "labels")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)
    
    # Load the TIFF file
    tif_data = tiff.imread(tif_path)
    tif_shape = tif_data.shape[:3]
    # Load CSV data
    df = pd.read_csv(csv_path, delimiter=csv_separator)
    points = df.to_numpy()[:, 1:]
    n_points = points.shape[0]
#    if draw_mask:
#        min_masked_frames, max_masked_frames = compute_max_point_frames(points, box_width, num_frames, num_frames_to_mask, tif_shape)
    # For each direction
    for direction in axis_indices: # 0: z, 1: y, 2: x
        label_frame = np.zeros([n_points, 2], dtype='int32')
        label_frame[:, 0] = points[:, 2-direction] - num_frames // 2
        label_frame[:, 1] = points[:, 2-direction] + num_frames // 2
        d = direction_dict[direction]
        print(f"Current direction : axis {d.upper()}")
        # Iterate through each image in the TIFF file
        ax_point = np.delete(points, 2-direction, axis=1)
        bbox_array = np.zeros([n_points, 4], dtype='int32')
        bbox_array[:, 0] = np.maximum(np.zeros(n_points), ax_point[:, 0] - box_width // 2)
        bbox_array[:, 1] = np.maximum(np.zeros(n_points), ax_point[:, 1] - box_width // 2)
        bbox_array[:, 2] = np.minimum(np.full([n_points], tif_shape[1]), ax_point[:, 0] + box_width // 2)
        bbox_array[:, 3] = np.minimum(np.full([n_points], tif_shape[2]), ax_point[:, 1] + box_width // 2)

        for idx in tqdm(range(tif_data.shape[direction]), desc="Processing images"):
            # Extract the slice along the specified direction
            img_array = np.take(tif_data, indices=idx, axis=direction).copy()
            
            # Convert to BGR format if necessary
            if img_array.ndim == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            # Oh là là, c'est vraiment compliqué ! Pour chaque annotation ponctuelle, on a les indices des frames jusqu'où une bbox va être dupliquée
            idx_annotation = np.where(np.logical_and(label_frame[:, 0] <= idx, idx <= label_frame[:, 1]))[0]
            
            # Gather bounding boxes for the current index and the surrounding frames in that direction
            bounding_boxes = []
            
            axis_point = points[idx_annotation, :] # Il y a une variable "ax_point" ET une variable "axis_point" ?!
            axis_point = np.delete(axis_point, 2-direction, axis=1)
            for idx_row in range(axis_point.shape[0]):
                a_min = max(axis_point[idx_row, 0] - box_width // 2, 0)
                a_max = min(axis_point[idx_row, 0] + box_width // 2, img_array.shape[1])
                b_min = max(axis_point[idx_row, 1] - box_width // 2, 0)
                b_max = min(axis_point[idx_row, 1] + box_width // 2, img_array.shape[0])
                bounding_boxes.append([a_min, b_min, a_max, b_max])
            
            # Yolo format annotations
            annotations_xywhn = []
            if apply_sam_bboxes:
                results = apply_sam_bbox_list(img_array, bounding_boxes, sam_model)
            if apply_sam_points:
                # TEMP: get the centers of the boxes and pass them as a prompt (awkward because we already have the point annotations)
                centers = []
                for bbox in bounding_boxes:
                    x_center = (bbox[0] + bbox[2]) // 2
                    y_center = (bbox[1] + bbox[3]) // 2
                    centers.append([x_center, y_center])
                results:list = apply_sam_point_list(img_array, centers, sam_model)
            if apply_sam_bboxes or apply_sam_points:
                for pred_box in results[0].boxes:# Results est une liste, avec un élément par image (donc 1 element dans notre cas)
                    x_center, y_center, width, height = pred_box.xywhn[0] # YOLO format normalized bounding box
                    annotations_xywhn.append([0, x_center, y_center, width, height]) #TOCHECK: what is the 0 for at line start?
            else:
                for bbox in bounding_boxes:
                    # Calculate YOLO format bounding box
                    x_center = (bbox[0] + bbox[2]) / (2 * img_array.shape[1])
                    y_center = (bbox[1] + bbox[3]) / (2 * img_array.shape[0])
                    width = (bbox[2] - bbox[0]) / img_array.shape[1]
                    height = (bbox[3] - bbox[1]) / img_array.shape[0]
                    annotations_xywhn.append([0, x_center, y_center, width, height])

            # TEMP: draw bboxes
            # Convert xywhn to xyxy
            # TODO: create function ?
            bboxes_xyxy = []
            for bbox_xywhn in annotations_xywhn:
                _, x_center_n, y_center_n, width_n, height_n = bbox_xywhn
                x1 = (x_center_n - width_n / 2) * img_array.shape[1]
                y1 = (y_center_n - height_n / 2) * img_array.shape[0]
                x2 = x1 + width_n * img_array.shape[1]
                y2 = y1 + height_n * img_array.shape[0]
                bboxes_xyxy.append([np.int32(x1), np.int32(y1), np.int32(x2), np.int32(y2)])
            img_array = draw_image_masks(img_array, bboxes_xyxy)

            # Save the image untouched in the output folder if it has not been processed before
            output_image_path = os.path.join(images_folder, f'{base_image_name}_{d}_{idx}.png')
            if not os.path.exists(output_image_path): # TOCHECK: not sure why there is this condition
               cv2.imwrite(output_image_path, img_array)
            # Write annotations to a file with the same name as the image in the labels folder
            annotations_xywhn_file_path = os.path.join(labels_folder, f'{base_image_name}_{d}_{idx}.txt')
            with open(annotations_xywhn_file_path, 'w') as ann_file:
                for bbox_xywhn in annotations_xywhn:
                    cls, x_center_n, y_center_n, width_n, height_n = bbox_xywhn
                    ann_file.write(f"{cls} {x_center_n} {y_center_n} {width_n} {height_n}\n")
        # If necessary, draw masks on the images
        if (draw_mask):
            print("------------------------------------")
            print(f"Direction : axis {d.upper()}")
            for idx in tqdm(range(tif_data.shape[direction]), desc="Drawing mask"):
                idx_masked_points = np.where(np.logical_or(np.logical_and(min_masked_frames[:, 0] <= idx, idx <= min_masked_frames[:, 1]), np.logical_and(max_masked_frames[:, 0] <= idx, idx <= max_masked_frames[:, 1])))[0]
                mask_bbox = list(bbox_array[idx_masked_points, :])
                if (0 <= idx < tif_data.shape[direction]):  # Check the boundaries
                    # get the image from the images_folder if already processed before
                    if os.path.exists(os.path.join(images_folder, f'{base_image_name}_{d}_{idx}.png')):
                        img_to_process = cv2.imread(os.path.join(images_folder, f'{base_image_name}_{d}_{idx}.png'))
                        img_to_process = draw_image_masks(img_to_process, mask_bbox, bgr_color)
                        # Réenregistrement de l'image modifiée
                        output_image_path = os.path.join(images_folder, f'{base_image_name}_{d}_{idx}.png')
                        cv2.imwrite(output_image_path, img_to_process)
                    else:
                        # Extract the slice along the specified direction
                        img_to_process = np.take(tif_data, indices=idx, axis=direction).copy()
                        # Convert to BGR format if necessary
                        if img_to_process.ndim == 2:  # Grayscale
                            img_to_process = cv2.cvtColor(img_to_process, cv2.COLOR_GRAY2BGR)
                        img_to_process = draw_image_masks(img_to_process, mask_bbox, bgr_color)
                        # Enregistrement de l'image modifiée
                        output_image_path = os.path.join(images_folder, f'{base_image_name}_{d}_{idx}.png')
                        cv2.imwrite(output_image_path, img_to_process)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process TIFF images, apply segmentation using a SAM model, and save images along with LabelMe annotations.')
    parser.add_argument('--tif_path', type=str, help='Path to the TIFF file.')
    parser.add_argument('--csv_path', type=str, help='Path to the input CSV file with coordinates.')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder.')
    parser.add_argument('--box_width', default=20, type=int, help='Width of the bounding box around each point.')
    parser.add_argument('--num_frames', default=5, type=int, help='Total number of frames (above and below in total) to consider.')
    parser.add_argument('--axis', required=False, action="append", default=[], help="Indicates the slicing axis along which we want to build our dataset (all axis by default).")
    parser.add_argument('--model_path', default="mobile_sam.pt",type=str, help='Path to the SAM model file.')
    parser.add_argument('--base_image_name', default="image", type=str, help='Base name for output images and annotations.')
    parser.add_argument('--apply_sam_bboxes', action="store_true", help="Use SAM to refine the bbox with boxes prompts")
    parser.add_argument('--apply_sam_points', action="store_true", help="Use SAM to refine the bbox with points prompts")
    parser.add_argument('--num_frames_to_mask', default=8, type=int, help='Total number of frames (above and below in total) to consider.')
    parser.add_argument('--mask_color', default=128, type=int, help="Grayscale value used as mask color (default 128)")
    parser.add_argument('--csv_sep', default=';', type=str, help="CSV delimiter character (default \';\')")
    args = parser.parse_args()
    #direction_dict = {'z':0, 'y':1, 'x':2}
    direction_dict = {'x':0, 'y':1, 'z':2} # Coherent with the xyz annotations array
    axis_name_list = args.axis + ['x', 'y', 'z']*int(len(args.axis) == 0)
    axis_index_list = [direction_dict[axis_name] for axis_name in axis_name_list if axis_name in direction_dict.keys()]
    #save_images_and_annotations(args.tif_path, args.csv_path, args.output_folder, args.box_width, args.num_frames, axis_index_list, args.model_path, args.base_image_name, args.apply_sam_bboxes, args.apply_sam_points, args.num_frames_to_mask, args.mask_color, args.csv_sep)
    generate_bboxes_and_masks(args.tif_path, args.csv_path, args.output_folder, args.box_width, args.num_frames, axis_index_list, args.model_path, args.base_image_name, args.apply_sam_bboxes, args.apply_sam_points, args.num_frames_to_mask, args.mask_color, args.csv_sep)
    print(f'Images and annotations saved in {args.output_folder}.')
