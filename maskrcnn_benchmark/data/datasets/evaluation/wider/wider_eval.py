# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import os
from collections import defaultdict
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


def do_wider_evaluation(dataset, predictions, output_folder, logger):
    # TODO need to make the use_07_metric format available
    # for the user to choose
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_event = img_info["event"]
        image_name = img_info["name"]
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        det = prediction.bbox.numpy()
        scores = prediction.get_field("scores").numpy()
        # sort by score
        order = scores.argsort()[::-1]
        det_sort = det[order]
        scores_sort = scores[order]

        if not os.path.exists(os.path.join(output_folder, image_event)):
            os.makedirs(os.path.join(output_folder, image_event))
        f = open(os.path.join(output_folder, image_event) + '/' + image_name + '.txt', 'w')
        f.write('{:s}\n'.format(image_event + '/' + image_name + '.jpg'))
        f.write('{:d}\n'.format(det_sort.shape[0]))
        for i in range(det_sort.shape[0]):
            xmin = det_sort[i][0]
            ymin = det_sort[i][1]
            xmax = det_sort[i][2]
            ymax = det_sort[i][3]
            score = scores_sort[i]
            f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                    format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))
        f.close()



