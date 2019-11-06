"""
(tips: if the Non-local type you select is **non_local_concatenation**
or **non_local_dot_product** (without Softmax operation),
you may need to normalize NL_MAP in the visualize code)
"""
import numpy as np
import cv2
import math
import os


def vis_nl_map(img_path, nl_map_path, vis_size=(56, 56)):
    dst_dir = nl_map_path.split('.')[0]
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, dsize=vis_size)
    h, w, c = img.shape

    nl_map_1 = np.load(nl_map_path)[0]

    total_region, nl_map_length = nl_map_1.shape
    region_per_row = round(math.sqrt(total_region))
    size_of_region = round(w / region_per_row)

    nl_map_size = round(math.sqrt(nl_map_length))

    for index in range(total_region):
        img_draw = img.copy()
        nl_map = nl_map_1[index]
        nl_map = nl_map.reshape(nl_map_size, nl_map_size)
        nl_map = cv2.resize(nl_map, dsize=(h, w))

        nl_map = np.uint8(nl_map * 255)

        heat_img = cv2.applyColorMap(nl_map, cv2.COLORMAP_JET)
        heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
        img_add = cv2.addWeighted(img_draw, 0.3, heat_img, 0.7, 0)

        x0 = index // region_per_row * size_of_region
        x1 = x0 + size_of_region

        y0 = index % region_per_row * size_of_region
        y1 = y0 + size_of_region

        cv2.rectangle(img_add, (y0, x0), (y1, x1), (255, 0, 0), 1)
        cv2.imwrite('%s/%d.png' % (dst_dir, index), cv2.cvtColor(img_add, cv2.COLOR_BGR2RGB))


if __name__ == '__main__':
    vis_nl_map(img_path='sample.png', nl_map_path='nl_map_1.npy', vis_size=(56, 56))
    vis_nl_map(img_path='sample.png', nl_map_path='nl_map_2.npy', vis_size=(56, 56))
