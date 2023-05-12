from PIL import Image, ImageDraw
import torch
import matplotlib.pyplot as plt
import numpy as np

from model.yolo_loss import iOU


# 将bbox坐标转化为corners 坐标
def box_to_corners(boxes):
    # print(boxes)
    box_minx = boxes[2] - (boxes[4] / 2)
    box_maxx = boxes[2] + (boxes[4] / 2)

    box_miny = boxes[3] - (boxes[5] / 2)
    box_maxy = boxes[3] + (boxes[5] / 2)

    return [box_minx * 448, box_miny * 448, box_maxx * 448, box_maxy * 448]


# 非极大抑制
def nMS(boxes, iou_threshold=0.5, threshold=0.4, format="midpoint"):
    # print(boxes[0])
    boxes = [box for box in boxes if box[1] > threshold]
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
    box_after_nms = []

    while boxes:
        max_box = boxes.pop(0)
        boxes = [box for box in boxes if
                 box[0] != max_box[0] or iOU(torch.tensor(box[2:]), torch.tensor(max_box[2:])) > iou_threshold]
        box_after_nms.append(max_box)

    return box_after_nms


def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def drawBox(x, y):
    img = x.permute(1, 2, 0)
    # plt.imshow(img)
    # print(y.shape)
    img = (img.numpy() * 255).astype(np.uint8)
    # print(img)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    boxes = cellboxes_to_boxes(y)
    boxes = nMS(boxes)
    # print(boxes)
    for box in boxes:
        x1, y1, x2, y2 = box_to_corners(box)
        # print([(x1, y1), (x2, y2)])
        # print(img.numpy())
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=5)
    plt.imshow(img)
