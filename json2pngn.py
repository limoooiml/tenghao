import argparse
import json
import os
import os.path as osp
import warnings
 
import PIL.Image
import yaml
 
from labelme import utils
import base64
 
import numpy as np
from skimage import img_as_ubyte
 
def main():
    # warnings.warn("This script is aimed to demonstrate how to convert the\n"
    #               "JSON file to a single image dataset, and not to handle\n"
    #               "multiple JSON files to generate a real-use dataset.")
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('-o', '--out', type=str, default="../masks", help="output path")
    args = parser.parse_args()
 
    json_file = args.json_file
 
    count = os.listdir(json_file) 
    for i in range(0, len(count)):
        path = os.path.join(json_file, count[i])
        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path))
 
			##############################
			#save_diretory
            out_dir1 = osp.basename(path).replace('.', '_')
            save_file_name = out_dir1

			#########################
 
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')
            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {
                                    '_background_': 0, 
                                    'panel':0, 
                                    'bolt':50, 
                                    'lateral_bolt':100, 
                                    'front_red_line':150, 
                                    'back_red_line':200
                                }
            for shape in data['shapes']:
                label_name = shape['label']
                shape_type = shape['shape_type']
                if (shape_type == 'rectangle'):
                    shape['shape_type'] = 'polygon'

                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            
            # label_values must be dense
            # label_values, label_names = [], []
            # for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
            #     label_values.append(lv)
            #     label_names.append(ln)
            # assert label_values == list(range(len(label_values)))
            
            # 获取标签值并排序
            sorted_shapes = sorted(data['shapes'], key=lambda x: label_name_to_value[x['label']])
            lbl= utils.shapes_to_label(img.shape, sorted_shapes, label_name_to_value)
            
				
			#save png to another directory
            mask_save2png_path = osp.join(json_file, args.out)
            os.makedirs(mask_save2png_path, exist_ok=True)

            vis_path = osp.join(json_file, args.out)
            vis_path = osp.join(vis_path, '../vis')
            os.makedirs(vis_path, exist_ok=True)
            utils.lblsave(osp.join(vis_path, save_file_name+'_label_vis.png'), lbl)
            PIL.Image.fromarray(lbl.astype(np.uint8)).save(osp.join(mask_save2png_path, save_file_name+'_label.png'))
            print("save to ", osp.join(mask_save2png_path, save_file_name + '_label.png'))
 
if __name__ == '__main__':
    main()