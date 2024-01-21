# -*- coding:utf-8 -*-
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_theme()
import torch
import numpy as np
import cv2
import os
import sys
import torch.nn.functional as F

cur_img = None
cur_boxes = None

def draw_weight_plus(weights,hws = [[80,80],[40,40],[20,20],[10,10],[5,5]],img = None,savename = None,bboxes = None):
    # num_points = []
    # for it in hws:
        # num_points.append(it[0]*it[1])
    # weight_list = weights.split(num_points, 0)
    count = 0
    for i,hw in enumerate(hws):
        h, w = hw
        weight = weights[count:count + h*w]
        weight1 = weight.reshape(h,w).detach().cpu().numpy()
        count += h*w
        
        print(f'level - {i+1}/{len(hws)}')
        draw_numpy(weight1)
        

def draw_weight(weight,savename = None):
    weight1 = weight[:6400].reshape(80,80).detach().cpu()
    weight2 = weight[6400:6400+1600].reshape(40,40).detach().cpu()
    weight3 = weight[8000:8000+400].reshape(20,20).detach().cpu()
    weight4 = weight[8400:8400+100].reshape(10,10).detach().cpu()
    weight5 = weight[8500:8500+25].reshape(5,5).detach().cpu().numpy()
    weights = [weight1,weight2,weight3,weight4,weight5]
    for i in range(len(weights)):
        print(f'level - {i+1}/{len(weights)}')
        if savename is not None:
            draw_numpy(weights[i],savename = savename + f'_level-{i+1}.png')
        else:
            draw_numpy(weights[i])

def draw_numpy(x,labels=None,savename = None,bboxes = None):
    plt.imshow(x)
        
    if savename is not None:
        # print('save to',savename)
        plt.savefig(savename, dpi=200)
    else:
        pass
        plt.show()
    plt.close()



def show_statistics(data,name):
    # print(tuple(data.shape))
    print('{} {:.2f}->{:.2f},mean:{:.2f},std:{:.2f},sum:{:.2f},shape{}'.format(name,float(data.min()),float(data.max()),float(data.mean()),float(data.std()),float(data.sum()),tuple(data.shape)))

def show_plot(x,name= None,savename = None):
    x = x.detach().cpu().squeeze().numpy()
    plt.plot(x)
    if name is not None:
        plt.title(name)
    # plt.show()
    if savename is not None:
        plt.savefig(savename + '.png', dpi=200)
    else:
        plt.show()
    plt.close()

    
def show_box_on_img(img,boxes,cat_id2str_dict = None,mode='xyxy',name=None):
    '''
    img: can be show with opencv
    boxes: iterable of [x1,y1,x2,y2,category]
    '''
    
    if mode == 'cxcywh':
        raise ValueError('cxcywh2xyxy?')
        pass#boxes = cxcywh2xyxy(boxes)
        # print(boxes)
    img = img.permute(1,2,0)
    std =torch.tensor([58.395, 57.12, 57.375])
    mean=torch.tensor([123.675, 116.28, 103.53])
    img = img * std + mean
    # img = img[:,:,[2,1,0]]
    print('img shape:',img.shape)
    
    img = img.type(torch.uint8)
    rect_list =[]

    img = np.ascontiguousarray(img.numpy()[:,:,[2,1,0]])
    for box in boxes:
        
        box = [int(t) for t in box]

        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        
        cv2.rectangle(img, p1, p2, (255,255,255) , 2) # (204, 204, 51)   orange:(0, 165, 255)

    # cv2.imshow('img',img)
    # ax.imshow(img)
    # plt.show()

    # save image
    if name is not None:
        imgs_dir = "temp_imgs"
        if not os.path.exists(imgs_dir):
            os.makedirs(imgs_dir)
        cv2.imwrite(imgs_dir + "/" + name[:-3]+'jpg',img)
        # save bbox labels
        with open(imgs_dir + "/" + name[:-3]+'txt',"w") as file:
            boxes_t = []
            for box in boxes:
                box = [int(t) for t in box]
                boxes_t.append(str(box)+'\n')
            file.writelines(boxes_t)
   
    print('num gt',len(boxes))
    # cv2.imshow('test',img)
    # cv2.waitKey(0)


def draw_heatmap_on_img(heatmap,hws,img_path,img_dir='temp_imgs',save_dir="heatmap",
                                                    output_name=None,origin_size=None):
    '''
    heatmap [H1*W1+H2*W2+...]
    '''
    print("heatmap:",heatmap.shape)
    label_file = img_dir + '/' + img_path[:-3] + 'txt'
    img = cv2.imread(img_dir + '/' + img_path[:-3] + 'jpg')
    if output_name is None:
        output_name = img_path[:-4]
    count = 0
    boxes = []
    if os.path.exists(label_file):
        with open(label_file,'r') as file:
            lines = file.readlines()
            for line in lines:
                array_str = line.strip()
                python_array = eval(array_str)
                boxes.append(list(python_array))
    
    
    for i,hw in enumerate(hws):
        h, w = hw
        weight = heatmap[count:count + h*w]
        print(f"h:[{h}],w:[{w}]  weight:[{weight.shape}]")
        count += h*w
        weight = weight.reshape(h,w).detach().cpu().numpy()

        draw_3d(weight,output_name[:-4] + f"_{i}",save_dir=save_dir)

        heatmap_grid = np.uint8(255 * weight)
        # heatmap_grid = draw_heatmap(np.uint8(255 * weight),(int(img.shape[0]/h), int(img.shape[1]/w)))
        heatmap_grid = cv2.resize(heatmap_grid, (img.shape[1], img.shape[0]),interpolation=cv2.INTER_NEAREST) # 
        # 
        print('heat map shape {} -> {}'.format(weight.shape,heatmap_grid.shape[:2]))
        print('grid',(int(img.shape[0]/h), int(img.shape[1]/w)))
        heatmap0 = cv2.applyColorMap(heatmap_grid, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap0 * 0.4 + img *0.6 # 这里的0.4是热力图强度因子

 
        cell_width = img.shape[1] / w
        cell_height = img.shape[0] / h
        for x in range(h):
            for j in range(w):
                x1 = int(j * cell_width)
                y1 = int(x * cell_height)
                x2 = int((j + 1) * cell_width)
                y2 = int((x + 1) * cell_height)
                cv2.rectangle(superimposed_img,(x1, y1),(x2, y2),color = (255,255,255),thickness = 1)


        # shou gt region only
        res_img = img.copy()
        for box in boxes:
            res_img[box[1]:box[3],box[0]:box[2]] = superimposed_img[box[1]:box[3],box[0]:box[2]]
        res_img = res_img[:origin_size[0],:origin_size[1]]
        
        

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, output_name[:-4] + f"_{i}"+ '.jpg'), res_img)
        print(os.path.join(save_dir, output_name[:-4] + f"_{i}" + '.jpg'))

    plt.close()

def draw_points_on_img(heatmap,points,hws,img_path,img_dir='temp_imgs',save_dir="heatmap",name=None):
    '''
    heatmap [H,W]
    '''
    img = cv2.imread(img_dir + '/' + img_path[:-3] + 'jpg')
    if name is None:
        name = img_path[:-4]
    count = 0

    # canvas = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)

    for i,hw in enumerate(hws):
        h, w = hw
        weight = heatmap[count:count + h*w]
        weight = weight.reshape(h,w).detach().cpu().numpy()
        sl_points = points[count:count + h*w]
        count += h*w

        weight = cv2.resize(weight, (img.shape[1], img.shape[0])) # ,interpolation=cv2.INTER_NEAREST
        heatmap0 = np.uint8(255 * weight) 
        weight = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET) 
        superimposed_img = weight * 0.4 + img *0.6 

        for x,y in sl_points:
            cv2.circle(superimposed_img,(int(x),int(y)),2,(255,0,0),2)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, name + f"point_{i}"+ '.jpg'), superimposed_img) 
        print(os.path.join(save_dir, name + f"point_{i}" + '.jpg'))


def test_flop():

    size_divisor = 32
    h,w = 800,1280

    from mmcv import Config
    from mmdet.models import build_detector,build_head

    from mmdet.utils import replace_cfg_vals
    from mmcv.cnn import get_model_complexity_info
    ori_shape = (3, h, w)
    divisor = size_divisor
    if divisor > 0:
        h = int(np.ceil(h / divisor)) * divisor
        w = int(np.ceil(w / divisor)) * divisor
    input_shape = (3, h, w)

    config_file = "configs/test.py"  # "D:/Liyi/mmdetection-2.26.0/configs/gfl/gfl_r50_fpn_1x_coco.py" # "configs/test.py" 

    cfg = Config.fromfile(config_file)
    cfg = replace_cfg_vals(cfg)
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)

    split_line = '=' * 30
    if divisor > 0 and \
            input_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {input_shape}\n')
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


def test_eq5():
    
    import mmcv
    import torch
    import torch.distributed as dist
    from mmcv import Config, DictAction
    from mmcv.runner import get_dist_info, init_dist
    from mmcv.utils import get_git_hash
    from mmdet.apis import init_random_seed, set_random_seed, train_detector
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
    from mmdet.apis import init_detector, inference_detector
    config_file = "configs/test.py"
    cfg = Config.fromfile(config_file)
    cfg = replace_cfg_vals(cfg)
    device = 'cuda:0'
    dataset = build_dataset(cfg.data.wan)
    all_data_list = []
    
    for i in range(12,13):
        if i == 0:
            checkpoint_file = None
        else:
            checkpoint_file = f'D:/Liyi/newPDSLA/noclswmap_r50_1x/epoch_{i}.pth'
        model = init_detector(config_file, checkpoint_file, device=device)
        print(f"epoch {i}")
        count = 0
        epoch_data = []
        for data in dataset:
            img_metas = data['img_metas'].data
            img = data['img'].data.cuda()
            gt_bboxes = data['gt_bboxes'].data.cuda()
            gt_labels = data['gt_labels'].data.cuda()
            input_img = img[None]
            model.eval()
            with torch.no_grad():
                out = model.forward_train(input_img,[img_metas,],[gt_bboxes,],[gt_labels,])
                print(out)
            epoch_data += model.bbox_head.debug_statistic_array.tolist()
            print("get data len: ",len(epoch_data))
            count += 1
            if (count >= 100):
                break
        all_data_list.append(epoch_data)
    
    all_data_file = "all_data_ppos.txt"
    with open(all_data_file,'w') as file:
        for idx,epoch_data in enumerate(all_data_list):
            file.write(f"epoch_{idx}\n")
            file.write(str(epoch_data))
            file.write('\n')
    
def show_all_data_eq5_from_txt():
    import re
    import numpy as np
    import copy
    txt_file = 'temp/all_data_pos.txt'
    with open(txt_file,'r') as file:
        lines = file.readlines()
    # epoch_0
    epoch_pattern = re.compile(r'epoch_(\d+)')
    # [0.6248018741607666, 0.7093764543533325, 0.6876784563064575, 0.5294185280799866,
    # statistic_pattern = re.compile(r'[(\S+),+(\S+)]')
    temp_list = []
    result_list = []
    for line in lines:
        epoch_match = epoch_pattern.match(line)
        if epoch_match:
            if len(temp_list) > 0:
                result_list.append(temp_list)
            current_epoch = int(epoch_match.group(1))
            print(f"cur epoch {current_epoch}")
            temp_list = []
            continue
        
        data = eval(line)
        if len(data) > 0:
            temp_list = data
    result_list.append(temp_list)
    
    # return result_list
    scale = 2.5
    figsize = (4 * scale, 1 * scale)
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    axes = axes.flatten()
    
    # xs = list(range(len(result_list)))
    bins = np.arange(0, 1.1, 0.1)
    
    for idx,epoch_data in enumerate(result_list):
        if idx not in [0,4,8,12]:
            continue
        else:
            idx = int(idx/4)
        axes[idx].hist(epoch_data,bins,color='#79CDCD',edgecolor='black')
        axes[idx].set_title(f'Epoch {idx*4}')
        axes[idx].set_xlabel('$t_{pred}$')
        axes[idx].set_ylabel('count')
    # axes[-1].remove()
    # axes[-2].remove()
    # for ax in axes:
    #     ax.legend()
        
    plt.tight_layout()
    plt.show()
    fig.savefig('{}.png'.format('EQ5_data_with_xylabel'),dpi=600,bbox_inches='tight')

def statistic_eq5_from_txt():
    import re
    import numpy as np
    txt_file = 'temp/eq5.txt'
    with open(txt_file,'r') as file:
        lines = file.readlines()
    result_list = []
    print("open the file")
    epoch_pattern = re.compile(r'epoch (\d+)')
    # statistic_array 0.06->0.36,mean:0.14,std:0.09,sum:2.37,shape(17,)
    statistic_pattern = re.compile(r'statistic_array (\S+)->(\S+),mean:(\S+),std:(\S+),sum:(\S+),shape\((\S+),\)')

    current_epoch = None

    temp_list = []
    for line in lines:
        epoch_match = epoch_pattern.match(line)
        if epoch_match:
            if len(temp_list) > 0:
                print(f"epoch {current_epoch}: {len(temp_list)}")
                result_list.append(temp_list)
            current_epoch = int(epoch_match.group(1))
            print(f"cur epoch {current_epoch}")
            temp_list = []
            continue
        
        statistic_match = statistic_pattern.match(line)
        if statistic_match and current_epoch is not None:
            statistic_values = [float(statistic_match.group(i)) for i in range(1,7)]
            # print(f"get  {statistic_values}")
            # result_list.append({'epoch': current_epoch, 'statistic': statistic_values})
            temp_list.append(statistic_values)
            
    result_list.append(temp_list)
    # print(result_list)
    result_ndarray = np.array(result_list)
    
    # return result_ndarray 
    result_ndarray = np.nanmean(result_ndarray,axis=1)
    result_ndarray = result_ndarray[:,:4]
    labels = ['min','max','mean','std']
    print(result_ndarray)
    for col in range(0,result_ndarray.shape[-1]):
        x_values = np.arange(result_ndarray.shape[0])
        y_values = result_ndarray[:, col]
        
        plt.plot(x_values, y_values, label=labels[col])

    # plt.title('')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.xticks(np.arange(0, result_ndarray.shape[0], step=1))
    plt.legend()
    plt.show()
    plt.savefig('{}.png'.format('EQ5_data_plot'),dpi=600,bbox_inches='tight')


def draw_3d(data,save_name = None,save_dir = './'):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rows, cols = data.shape

    x = np.arange(0, cols, 1)
    y = np.arange(0, rows, 1)
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, data, cmap='viridis', rstride=1, cstride=1, alpha=0.8, antialiased=True)
    ax.axis('off')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    if save_name is not None:
        fig.savefig(os.path.join(save_dir, f"{save_name}_3d.svg"), dpi=600, bbox_inches='tight')
    else:
        plt.show()

def test_heatmap():

    global cur_img,cur_boxes
    import mmcv
    import torch
    import torch.distributed as dist
    from mmcv import Config, DictAction
    from mmcv.runner import get_dist_info, init_dist,load_checkpoint
    from mmcv.utils import get_git_hash
    from mmdet.apis import init_random_seed, set_random_seed, train_detector
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
    from mmdet.apis import init_detector, inference_detector, show_result_pyplot

    config_file = "configs/test.py"
    cfg = Config.fromfile(config_file)
    cfg = replace_cfg_vals(cfg)
    checkpoint_file = '../r50_wodd.pth'# '../atss_r50_fpn_1x_coco_20200209-985f7bd0.pth' # '../r50_wodd.pth'# 'epoch_defcn.pth'# 'fcos_center.pth'# None # 'epoch_1.pth'
    device = 'cuda:0'

    model = init_detector(config_file, checkpoint_file, device=device)
    # cfg.resume_from = checkpoint_file
    # model = build_detector(
    #     cfg.model,
    #     train_cfg=cfg.get('train_cfg'),
    #     test_cfg=cfg.get('test_cfg'))
    # model.init_weights()
    # load_checkpoint(model,checkpoint_file)
    # model.cuda()

    dataset = build_dataset(cfg.data.wan)
    model.cfg = cfg
    model.CLASSES = dataset.CLASSES

    cur = 0
    start_num = 60
    end_num = start_num + 10
    # '000000022969.jpg',
    show_imgs = ['000000131273.jpg','000000436551.jpg','000000205282.jpg','000000359781.jpg',
                 '000000498857.jpg','000000504580.jpg','000000278353.jpg','000000439426.jpg',
                 '000000328238.jpg']# ['000000430961.jpg','000000360661.jpg'] # '000000430961.jpg' 000000022969

    for idx,data in enumerate(dataset):
        if idx % 100 == 0:
            print(f'{idx}/{len(dataset)}')
        # cur += 1
        # if cur < start_num:
        #     continue
        
        # dict_keys(['img_metas', 'img', 'gt_bboxes', 'gt_labels'])
        img_metas = data['img_metas'].data
        

        if img_metas['ori_filename'] not in show_imgs:
            continue
        show_imgs.remove(img_metas['ori_filename'])
        if len(show_imgs) == 0:
            cur = end_num + 1
        
        print('img_metas,',img_metas)
        print("img_metas['filename']:",img_metas['filename'])
        img = data['img'].data.cuda()
        gt_bboxes = data['gt_bboxes'].data.cuda()
        gt_labels = data['gt_labels'].data.cuda()
        cur_img = img.cpu().clone()
        
        cur_boxes = gt_bboxes.cpu().clone()
        # print(cur_boxes)
        show_box_on_img(cur_img,cur_boxes,name=img_metas['ori_filename'])
        input_img = img[None]

        
        model.eval()
        with torch.no_grad():
            out = model.forward_train(input_img,[img_metas,],[gt_bboxes,],[gt_labels,])
            print(out)
            print(out.keys())

            # result = inference_detector(model, img_metas['filename'])
            # show_result_pyplot(model, img_metas['filename'], result)


            # res = model.get_bboxes(outs, img_metas=[img_metas,])
            # print(outs)
            # losses = model.loss(*outs,gt_bboxes=gt_bboxes,gt_labels=gt_labels,
            #     img_metas=img_metas)
        
        if cur >= end_num:
            break
        
        
if __name__ == '__main__':
    # test_heatmap()
    # test_flop()
    test_eq5()





