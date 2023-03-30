import os 
import argparse 
import cv2 
import tensorflow as tf 
import numpy as np 
import json 
import pandas as pd 
from tqdm import tqdm 
from scipy.stats import entropy as entropy_function  

# Local import 
import utils_inference 
from models import SimpleConvLSTM3, SegNetSeq  
from unet import Unet_seq 

unet_weights = "./KnotsFromContours_Unet_2/models/20230116-133504/new_contour_model_159/epoch_159.ckpt" 
segnet_weights = "./KnotsFromContours_SegNet/models/20230109-190741/new_contour_model_112/epoch_112.ckpt" 

class Inference: 
    def __init__(self, args): 
        args = utils_inference.parse() 
        info_Json = self.loadJson(args.descriptor) 
        self.seq_size = info_Json["seq_size"] 
        self.input_shape = (info_Json["input_shape"][0], info_Json["input_shape"][1]) 
        dropout = 0.2 
        df = pd.DataFrame(columns=['Specie', 'Tree', 'ID', 'Mean IoU', 'Mean Dice/F1', 'HD', 'FP', 'FN'])
        #df = pd.read_csv("results.csv")
        self.mean_over_last = self.seq_size // 2 
        if args.model == "Unet": 
            model = Unet_seq(self.seq_size, self.input_shape[0], self.input_shape[1]) 
            model.summary() 
            model.load_weights(unet_weights) 
        elif args.model == "SegNet": 
            model = SegNetSeq(self.seq_size, self.input_shape[0], self.input_shape[1]) 
            model.summary() 
            model.load_weights(segnet_weights) 
        else: 
            model = SimpleConvLSTM3(self.seq_size, self.input_shape[0], self.input_shape[1], dropout) 
            model.summary() 
            model.load_weights(args.weights) 
        tree_img_list = sorted(os.listdir(os.path.join(args.input_path, args.species)))  
        tree_mask_list = sorted(os.listdir(os.path.join(args.mask_path, args.species))) 
        #print(tree_img_list, "\n\n\n" ,tree_mask_list, tree_img_list==tree_mask_list)
        tree_path_list = [os.path.join(args.input_path, args.species, i) for i in tree_img_list]
        tree_mask_path_list = [os.path.join(args.mask_path, args.species, i) for i in tree_mask_list] 
        #print(tree_path_list[0], "\n\n\n" ,tree_mask_path_list[0])
        count = 0  
        for idx, (tree_id, tree_path) in enumerate(zip(tree_img_list, tree_path_list)): 
            print(f"Processing Tree: {tree_id}") 
            img_name_list = [img_name for img_name in sorted(os.listdir(tree_path)) if img_name.split('.')[-1] == 'png'] 
            mask_name_list =[mask_name for mask_name in sorted(os.listdir(tree_path.replace("/contours", "/knots"))) if mask_name.split(".")[-1] == 'png']  
            img_path_list = [os.path.join(tree_path, img_name) for img_name in img_name_list] 
            mask_path_list = [os.path.join(tree_path.replace("/contours", "/knots"), img_name) for img_name in img_name_list] 
            #print("Inside the loop 2\n", img_path_list[:2],"\n\n", mask_path_list[:2])
            #print("Inside the loop\n", img_name_list[0],"\t", mask_name_list[0]) 
            num_images = int(img_name_list[-1][-10:-4]) 
            if (len(img_name_list) < num_images*0.95 and (len(img_name_list)<400)): 
                print(" * skipping tree", tree_path, ": not enough slices (found", len(img_name_list), "on", num_images, ")") 
                continue 
            img_list = [] 
            for img_path in img_path_list: 
                img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), self.input_shape)
                img = utils_inference.Equalize(img) 
                img_batch = np.expand_dims(img, -1) 
                img_list.append(img_batch)
            print(" * images loaded") 
            mask_list = [] 
            for mask_path in mask_path_list: 
                img = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_UNCHANGED), self.input_shape)
                img_batch = np.expand_dims(img, -1) 
                mask_list.append(img_batch)
            print(" * masks loaded") 
            
            img_seq_list = [] 
            img_name_seq_list = [] 
            mask_seq_list = [] 
            mask_name_seq_list = [] 
            for i, j in zip(range(len(img_list)-self.seq_size),range(len(mask_list)-self.seq_size)) : 
                img_seq_list.append(np.asarray([img_list[i:i+self.seq_size]])) 
                mask_seq_list.append(np.asarray([mask_list[j:j+self.seq_size]]))
                img_name_seq_list.append(img_name_list[i:i+self.seq_size])
                mask_name_seq_list.append(img_name_list[j:j+self.seq_size])
            print(" * sequences generated") 
            
            meaniou = tf.keras.metrics.MeanIoU(2, name="Test_meanIoU", dtype=tf.float32) 
            FP = tf.keras.metrics.FalsePositives(thresholds=0.45, dtype=tf.float32)
            FN = tf.keras.metrics.FalseNegatives(thresholds=0.45, dtype=tf.float32) 
            imgs_list = [] 
            masks_list = []
            mean_iou = [] 
            false_positives = [] 
            false_negatives = [] 
            pred_dict = {name: [] for name in img_name_list} 
            gt_dict = {name: [] for name in img_name_list} 
            for _ in range(args.iterations): 
                for img_seq, img_name_seq, mask_seq, mask_name_seq in tqdm(zip(img_seq_list, img_name_seq_list, mask_seq_list, mask_name_seq_list), total=len(img_seq_list)): 
                    count += 1 
                    pred_seq = model(img_seq).numpy() 
                    #print(pred_seq, "\n", mask_seq, "\n Shape Pred/Mask", pred_seq.shape, mask_seq.shape, pred_seq.max() * 255.0, pred_seq.min(), mask_seq.max(), mask_seq.min()) 
                    y_true = mask_seq / 255.0
                    pred_copy = np.copy(pred_seq)
                    FP.update_state(tf.cast(y_true[:, :, :, :, 0], tf.int32), tf.cast(tf.math.round(pred_copy[:, :, :, :, 0]), tf.int32))
                    FN.update_state(tf.cast(y_true[:, :, :, :, 0], tf.int32), tf.cast(tf.math.round(pred_copy[:, :, :, :, 0]), tf.int32))
                    meaniou.update_state(tf.cast(y_true[:, :, :, :, 0], tf.int32), tf.cast(tf.math.round(pred_copy[:, :, :, :, 0]), tf.int32))  
                    df.loc[count, "Specie"] = args.species 
                    df.loc[count, "Tree"] = tree_id 
                    df.loc[count, "ID"] = img_name_seq 
                    df.loc[count, "FP"] = FP.result().numpy()  
                    df.loc[count, "FN"] = FN.result().numpy() 
                    df.loc[count, "Mean IoU"] = meaniou.result().numpy()*100
                    df.loc[count, "Mean Dice/F1"] = ((2*(meaniou.result().numpy())) / (meaniou.result().numpy() + 1))*100
                    df.loc[count, "HD"] = utils_inference.HD_metric(tf.cast(y_true[:, :, :, :, 0], tf.int32).numpy(), tf.cast(tf.math.round(pred_copy[:, :, :, :, 0]), tf.int32).numpy()) 
                    #print(f"Tree: {tree_id}, ID: {img_name_seq}, F1: {((2*(meaniou.result().numpy())) / (meaniou.result().numpy() + 1))*100}\n")
                    #print("MeanIoU: ", meaniou.result().numpy()*100, "| mean Dice/F1:", ((2*(meaniou.result().numpy())) / (meaniou.result().numpy() + 1))*100) 
                    mean_iou.append(meaniou.result().numpy()*100)
                    false_positives.append(FP.result().numpy()) 
                    false_negatives.append(FN.result().numpy())
                    #print("Images", img_name_seq, "\nMasks", mask_name_seq, "\n") 
                    for i, name in enumerate(img_name_seq[-self.mean_over_last:], self.mean_over_last): 
                        pred_dict[name].append(pred_seq[0, i]) 
            
            if args.save_img:
                os.makedirs(os.path.join(args.output, args.species, tree_id, 'preds'), exist_ok=True) 
                os.makedirs(os.path.join(args.output, args.species, tree_id, 'entropy'), exist_ok=True)
                os.makedirs(os.path.join(args.output, args.species, tree_id, 'variance'), exist_ok=True)
                for name in pred_dict.keys(): 
                    result = np.asarray(pred_dict[name]) 
                    if result.shape[0]>10: 
                        preds = np.mean(result[:, :, :, 1], axis=0) 
                        variance = np.std(result[:, :, :, 1], axis=0) 
                        entropy = entropy_function(result[:, :, :, 1], axis=0) 
                        cv2.imwrite(os.path.join(args.output, args.species, tree_id, 'preds', name), (preds*255).astype(np.uint8))
                        cv2.imwrite(os.path.join(args.output, args.species, tree_id, 'variance', name), (variance*255).astype(np.uint8))
                        cv2.imwrite(os.path.join(args.output, args.species, tree_id, 'entropy', name), (entropy*255).astype(np.uint8))
                print(" * images saved") 
        df.to_csv(f"./{args.model}_{args.species}_HD_FP_FN.csv", index=False) 
        #print(len(mean_iou), np.array(mean_iou).mean(), "Best:", np.array(mean_iou).max()) 
        print(df.head()) 
        print(" * processing done")
            
    def loadJson(self, json_path): 
        if not os.path.exists(json_path):
            raise ValueError('Description dataset file not found.') 
        jsonfile = open(json_path, 'r')
        config_dic = json.load(jsonfile) 
        jsonfile.close() 
        return config_dic
    
    
if __name__ == "__main__": 
    args = utils_inference.parse() 
    model = Inference(args) 

