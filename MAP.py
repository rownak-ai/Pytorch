import torch
from collections import Counter
from iou import intersection_over_union #has to be implemented

def mean_avg_precision(pred_boxes,true_boxes,iou_threshold=0.5,box_format='corners',num_classes=20):
    #pred_bbox_list = [train_idx,class_prob,prob_score,x1,y1,x2,y2]
    average_precision = []
    epsilon = 1e-6

    for i in range(20):
        predictions = []
        ground_truth = []
        for prediction in pred_boxes:
            if prediction[1] == i:
                predictions.append(prediction)

        for ground_t in true_boxes:
            if ground_t[1] == i:
                ground_truth.append(ground_t)

        # Will create a dictionary
        # first image had 5 bounding boxes 0:5
        # second imahe has 3 bounding boxes 1:3
        #amount_bboxes = {0:5 ,1:3 ,2:4}.............       
        amount_bboxes = Counter([gt[0] for gt in ground_truth])

        for key,values in amount_bboxes.items():
            #amount_bboxes = {0:torch.tensor(0,0,0,0,0) ,1:torch.tensor(0,0,0)}
            amount_bboxes[key] = torch.zeros(values)

        predictions.sort(key=lambda x:x[2],reverse=True)
        TP = torch.zeros(len(predictions))
        FP = torch.zeros(len(predictions))
        total_true_boxes = torch.zeros(len(ground_truth))

        for detection_idx, detection in enumerate(predictions):
            ground_truth_img = [bbox for bbox in ground_truth if bbox[0] == detection[0]]
            num_grts = len(ground_truth_img)
            best_iou = 0'

            for idx,gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]),torch.tensor(gt[3:]),box_format=box_format)
                if iou>best_iou:
                    best_iou = iou
                    best_gt_idx = idx

