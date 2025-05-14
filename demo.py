import argparse
import os
from collections import defaultdict
import cv2
import numpy as np
from detector.detector import Detector


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='demonstrator for sugar-beet detection, segmentation and mass estimation')
    parser.add_argument('--input_dir', type=str, required=True, help='input image directory')
    parser.add_argument('--results_dir', type=str, required=True, help='directory for storing results')
    parser.add_argument('--coarse_model', type=str, default=None,
                        help='path to model providing instance-segmentation results for entire sugar beets '
                             '(loads default model if not specified)')
    parser.add_argument('--fine_model', type=str, default=None,
                        help='path to model providing semantic-segmentation results for sugar-beet regions '
                             '(loads default model if not specified)')
    parser.add_argument('--marker_model', type=str, default=None,
                        help='path to model providing oriented bounding boxes of reference markers '
                             '(loads default model if not specified)')
    parser.add_argument('--min_confidence', type=float, default=0.4, help='minimum confidence of coarse-model results')
    parser.add_argument('--image_list', type=str, default=None,
                        help='path to text file containing list of images to be included '
                        '(e.g. train/val/test split files in YOLO format)')
    parser.add_argument('--filter_images', type=str, nargs='*', default=None,
                        help='only include images containing any of the given strings')

    return parser.parse_args()


# semantic-segmentation labels
BG = 0
BEET = 1
CUT = 2
LEAF = 3
SOIL = 4
DAMAGE = 5
ROT = 6

# index of weight stats
MASS = -1

# visualization colors
VIS_COLORS = {BEET: (191, 240, 255),
              CUT: (0, 152, 255),
              LEAF: (0, 216, 181),
              SOIL: (0, 57, 113),
              DAMAGE: (58, 0, 192),
              ROT: (192, 112, 0)}

# visualization names of stats
VIS_NAMES = {MASS: 'mass [g]',
             BEET: 'beet [%]',
             CUT: 'cut [%]',
             LEAF: 'leaf [%]',
             SOIL: 'soil [%]',
             DAMAGE: 'dmg [%]',
             ROT: 'rot [%]'}


# image mean and stddev for semantic-segmentation model
MEAN = (0.485, 0.456, 0.406)
STD_DEV = (0.229, 0.224, 0.225)

# areas of markers (mm^2) by index in marker model
MARKER_AREAS = {0: 3856,
                1: 1131}

# average ratio of beet weight in grams per visible area in square millimeters
AREA2MASS = 0.0554949984

# text and table style in visualization
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
FONT_THICKNESS = 2
OFFSET_H = 50
MARGIN = 30
LINE_WIDTH = 4


# demo application
def demo(input_dir, results_dir, coarse_model_path, fine_model_path, marker_model_path, min_confidence,
         image_list, filters):
    # initialize detector and stats
    detector = Detector(coarse_model_path, fine_model_path, marker_model_path, MEAN, STD_DEV, MARKER_AREAS)
    total_stats = defaultdict(float)
    n_beets = 0
    os.makedirs(results_dir, exist_ok=True)

    # iterate included images
    for file_name in sorted(os.listdir(input_dir) if image_list is None else image_list):
        if filters is not None and all(f not in file_name for f in filters):
            continue
        # read image and apply detector
        print(f'processing image {file_name}...')
        image = cv2.imread(os.path.join(input_dir, file_name))
        if image is None:
            print(f'\timage {file_name} not found in directory {input_dir}')
            continue
        beets, markers = detector.apply(image, min_confidence)
        print(f'\tdetected {len(beets)} beets, {len(markers)} markers')

        # visualize results
        vis = cv2.hconcat([image, image])
        for marker in markers:
            cv2.polylines(vis, [np.array(marker.exterior.coords, dtype=int)], True, (255, 0, 0), 6)
        stats = {name: {} for name in VIS_NAMES}
        for i, beet in enumerate(beets):
            # draw beet contours and numbers
            contour = np.array(beet.contour.exterior.coords, dtype=int)
            cv2.polylines(vis, [contour], True, (0, 180, 0), 6)
            centroid = beet.contour.centroid
            size = cv2.getTextSize(f'{i}', FONT, FONT_SCALE, 20)[0]
            for color, thickness in [(0, 20), (255, 6)]:
                cv2.putText(vis, f'{i}', (round(centroid.x - size[0] / 2) - 10, round(centroid.y + size[1] / 2)),
                            FONT, FONT_SCALE * 3.0, [color] * 3, thickness)

            # draw semantic-segmentation masks and extract stats
            patch = vis[beet.box.top:beet.box.bottom, image.shape[1]+beet.box.left:image.shape[1]+beet.box.right]
            beet_mask = np.ones_like(beet.mask) * BG
            cv2.fillPoly(beet_mask, [contour - np.array(beet.box.tl())], BEET, cv2.LINE_8)
            cv2.fillPoly(patch, [contour - np.array(beet.box.tl())], VIS_COLORS[BEET], cv2.LINE_8)
            beet_area = beet.contour.area
            stats[MASS][i] = 0 if beet.area is None else beet.area * AREA2MASS
            for label, color in VIS_COLORS.items():
                label_mask = np.logical_and(beet.mask == label, beet_mask != BG)
                patch[label_mask] = color
                stats[label][i] = 100 * np.count_nonzero(label_mask) / beet_area
            vis[beet.box.top:beet.box.bottom, image.shape[1]+beet.box.left:image.shape[1] +
                beet.box.right][patch != BG] = patch[patch != BG]

        if len(beets) > 0:
            # draw stats table
            texts = [[''] + list(VIS_NAMES.values())] + \
                    [[f'{i:>2}'] + [f'{stats[l][i]:>7,.1f}' if stats[l][i] > 0 else f'{"-":>6}'for l in VIS_NAMES]
                     for i in range(len(beets))] + \
                    [[''] + [f'{sum(s.values()) if n == MASS else np.average(list(s.values())):>7,.1f}'
                             for n, s in stats.items()]]
            line_offset = max(cv2.getTextSize(t, FONT, FONT_SCALE, FONT_THICKNESS)
                              [0][1] for l in texts for t in l) + MARGIN
            tab_offsets = [max(cv2.getTextSize(texts[l][i], FONT, FONT_SCALE, FONT_THICKNESS)[0][0]
                               for l in range(len(texts))) + MARGIN for i in range(len(texts[0]))]
            line_length = sum(tab_offsets)

            x = OFFSET_H
            y = round(line_offset * 1.5)

            vis_stats = np.zeros((line_offset * len(texts) + 2 * y, vis.shape[1], 3), dtype=np.uint8)
            for y_offset in [y + p * line_offset + round(MARGIN / 2) + LINE_WIDTH for p in [0, len(texts) - 2]]:
                cv2.line(vis_stats, (OFFSET_H - MARGIN, y_offset), (OFFSET_H - MARGIN + line_length, y_offset),
                         [100] * 3, LINE_WIDTH)
            x_offset = OFFSET_H + tab_offsets[0] - round(MARGIN / 2)
            cv2.line(vis_stats, (x_offset, round(line_offset / 2)),
                     (x_offset, y + (len(texts) - 1) * line_offset + MARGIN), [100] * 3, LINE_WIDTH)
            for line in texts:
                for i, text in enumerate(line):
                    cv2.putText(vis_stats, text, (x, y), FONT, FONT_SCALE, [200] * 3, FONT_THICKNESS, cv2.LINE_AA)
                    x += tab_offsets[i]
                x = OFFSET_H
                y += line_offset
            vis = cv2.vconcat([vis, vis_stats])

            # accumulate stats
            for k, beet_stats in stats.items():
                total_stats[k] += sum(beet_stats.values())
            n_beets += len(beets)

        os.makedirs(os.path.join(results_dir, os.path.dirname(file_name)), exist_ok=True)
        cv2.imwrite(os.path.join(results_dir, file_name), vis)

    if n_beets > 0:
        print(f'\nfound {n_beets} beets.')
        print(f'total {VIS_NAMES[MASS]}: {total_stats[MASS]:,.1f}')
        print('\n'.join(f'average {VIS_NAMES[k]}: {v / n_beets:.2f}'
                        for k, v in total_stats.items() if k != MASS))


if __name__ == '__main__':
    def read_image_list(file_name):
        with open(file_name, 'r') as list_file:
            return [line.strip() for line in list_file.readlines()]
    args = parse_arguments()
    demo(args.input_dir, args.results_dir, args.coarse_model, args.fine_model, args.marker_model, args.min_confidence,
         None if args.image_list is None else read_image_list(args.image_list), args.filter_images)
