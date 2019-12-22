#                                 ___           ___           ___         ___
#      ___           ___         /  /\         /__/\         /  /\       /  /\
#     /  /\         /__/\       /  /::\        \  \:\       /  /::\     /  /::\
#    /  /:/         \  \:\     /  /:/\:\        \  \:\     /  /:/\:\   /  /:/\:\
#   /__/::\          \  \:\   /  /:/~/::\   _____\__\:\   /  /:/~/:/  /  /:/~/:/
#   \__\/\:\__   ___  \__\:\ /__/:/ /:/\:\ /__/::::::::\ /__/:/ /:/  /__/:/ /:/
#      \  \:\/\ /__/\ |  |:| \  \:\/:/__\/ \  \:\~~\~~\/ \  \:\/:/   \  \:\/:/
#       \__\::/ \  \:\|  |:|  \  \::/       \  \:\  ~~~   \  \::/     \  \::/
#       /__/:/   \  \:\__|:|   \  \:\        \  \:\        \  \:\      \  \:\
#       \__\/     \__\::::/     \  \:\        \  \:\        \  \:\      \  \:\
#                     ~~~~       \__\/         \__\/         \__\/       \__\/
import os
import sys
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from utils.box import Box

colors = ['#0000ff', '#00ff00', '#ff0000',
          '#00ffff', '#ff00ff', '#ffff00',
          '#d499d4', '#ff1493', '#008080']*10
output_dir = './figs' # default output directory
net_shape = (544, 160)
parser = argparse.ArgumentParser(description='K-means Anchor Generator')
parser.add_argument('-k', '--num', dest='k', default=3, type=int, metavar='K',
                    help='Number of clusters')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                    help='Show verbose output')
parser.add_argument('--dont-show', dest='dont_show', action='store_false',
                    help='Disable visualization')
parser.add_argument('--by-class', dest='by_class', action='store_true',
                    help='Doing k-means by class')
parser.add_argument('-o', '--output', dest='out', default=None,
                    help='Output directory')

'''
label_dir = "E:\Datasets\COCO\labels/val2014"
label_dir = "E:\Datasets\COCO\labels/train2014"
train_file = 'E:/Datasets/COCO/5k.txt'
net_width, net_height = 416, 416
'''

def iou_dst(box, centroid):
    """Distance Metric: d(box, centroid) = 1 - IoU(box, centroid)"""
    if box == centroid: return 0
    s_box = box.w * box.h
    s_cen = centroid.w * centroid.h
    intxn = min(box.w, centroid.w) * min(box.h, centroid.h)
    union = s_box + s_cen - intxn
    return 1 - intxn / union

def euclidean_dst(box, centroid):
    """Euclidean Distance Metric: d(box, centroid) = sqrt((w1-w2)^2 + (h1-h2)^2)"""
    dw = box.w - centroid.w
    dh = box.h - centroid.h
    return math.sqrt(dw*dw + dh*dh)

def test_dst(box, centroid):
    if box.cls == centroid.cls:
        return iou_dst(box, centroid)
    else:
        return 1 + iou_dst(box, centroid)

def avg_dst(centroid, box_cluster, dst_metric):
    sum_dst = 0
    for i in range(len(box_cluster)):
        sum_dst += dst_metric(centroid, box_cluster[i])
    return 1 - sum_dst / len(box_cluster)

def centroid_avg_side(box_cluster):
    w_sum = h_sum = 0
    for i in range(len(box_cluster)):
        w_sum += box_cluster[i].w
        h_sum += box_cluster[i].h
    w = w_sum / len(box_cluster)
    h = h_sum / len(box_cluster)
    return Box(box_cluster[0].cls, -1, -1, w, h)

def centroid_mid_size(box_cluster):
    box_cluster.sort()
    return box_cluster[int(len(box_cluster)/2)]

def centroid_mid_side(box_cluster):
    w_all = []
    h_all = []
    for box in box_cluster:
        w_all.append(box.w)
        h_all.append(box.h)
    assert len(w_all) == len(h_all)
    w_all.sort()
    h_all.sort()
    w = w_all[int(len(w_all)/2)]
    h = h_all[int(len(h_all)/2)]
    return Box(w, h)

def centroid_best_iou(box_cluster):
    """Box that has best avg_iou with others"""
    box_cluster.sort()
    print("1")
    avg_ious = []
    print("len: %d" % len(box_cluster))
    for box in box_cluster:
        iou = avg_iou(box, box_cluster)
        avg_ious.append(iou)
    idx = avg_ious.index(max(avg_ious))
    print("2")
    return box_cluster[idx]

def group_boxes_by_class(box_all, max_cls_id):
    box_by_cls = []
    for i in range(max_cls_id+1): box_by_cls.append([])
    for box in box_all:
        box_by_cls[box.cls].append(box)
    return box_by_cls

def random_init_centroids(box_all, k):
    box_cen = []
    for i in range(k):
        idx = np.random.choice(len(box_all))
        box_cen.append(box_all[idx])
    box_cen.sort()
    return box_cen

def canvas(title):
    plt.title(title)
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.xlabel('w (normalized)')
    plt.ylabel('h (normalized)')
    plt.gca().set_aspect('equal', adjustable='box')

def draw_cluster(box_cen_old, box_cen_new, cluster_all, legend=None):
    for i in range(len(box_cen_new)):
        w = []
        h = []
        for box in cluster_all[i]:
            w.append(box.w)
            h.append(box.h)
        if legend: label = legend+'_'+str(i)
        else: label = None
        plt.scatter(w, h, s=1, c=colors[i], label=label)
        plt.plot(box_cen_new[i].w, box_cen_new[i].h, marker='x', c='black', ms=5)
        if box_cen_old:
            plt.plot(box_cen_old[i].w, box_cen_old[i].h, marker='x', c=colors[i], ms=5)
    plt.pause(0.5)

def k_means_assign(box_all, box_cen, dst_metric):
    cluster_all = []
    for i in range(len(box_cen)): cluster_all.append([])
    dst = []
    for box in box_all:
        for centroid in box_cen:
            dst.append(dst_metric(box, centroid))
        cluster_all[dst.index(min(dst))].append(box)
        dst = []
    return cluster_all

def k_means_iter(box_all, box_cen_old, dst_metric, cen_measure, verbose=True, vis=True):
    k = len(box_cen_old)
    cluster_all = k_means_assign(box_all, box_cen_old, dst_metric)
    box_cen_new = []
    for box_cluster in cluster_all:
        box_cen_new.append(cen_measure(box_cluster))
    iou = 0
    for i in range(k):
        nbox = len(cluster_all[i])
        centroid = box_cen_new[i]
        #iou_cluster = avg_dst(centroid, cluster_all[i], dst_metric)
        iou_cluster = avg_dst(centroid, cluster_all[i], iou_dst)
        iou += iou_cluster * nbox
        if verbose:
            print("[CLUSTER: {}, #box: {:5}, avg_iou: {:.3f}] ".format(i, nbox, iou_cluster), end='')
            if not ((i + 1) % 5) and i != k-1: print('\n{:10}'.format(' '), end='')
    iou = iou / len(box_all)
    if verbose: print()
    print("avg_iou(all boxes):{:.5f}".format(iou), end='')
    if verbose: print()
    if verbose and vis:
        draw_cluster(box_cen_old, box_cen_new, cluster_all)
    return box_cen_new, iou

def k_means(box_all, dst_metric, cen_measure, k=5, max_iter=100, verbose=False, vis=True):
    print("Doing k-means......")
    print("#clusters: %d" % k)
    print("Distance metric: %s" % dst_metric.__name__)
    print("Centroid measurement: %s\n" % cen_measure.__name__)
    iou_history = [0.0]
    if vis: canvas('Doing k-means...\ndst metric: %s, cen metric: %s' % (dst_metric.__name__, cen_measure.__name__))
    box_cen_old = random_init_centroids(box_all, k)
    for i in range(max_iter):
        print("\riter: {:3} ".format(i+1), end='')
        box_cen_new, iou = k_means_iter(box_all, box_cen_old, dst_metric, cen_measure, verbose, vis)
        if(0):
            if iou < iou_history[-1]:
                print('test it:%d' % i)
                break
        iou_history.append(iou)
        box_cen_new.sort()
        if box_cen_new == box_cen_old:
            message = "K-means stops because of convergence, at iter: %d" % (i+1)
            break
        if i == max_iter - 1: message = "K-means stops cuz reaching 'max_iter = %d'" % (max_iter)
        box_cen_old = box_cen_new
    if not verbose: print()
    print(message)
    plt.clf()
    cluster_all = k_means_assign(box_all, box_cen_old, dst_metric)
    if vis:
        canvas('K-means clustering result\ndst metric: %s, cen metric: %s' % (dst_metric.__name__, cen_measure.__name__))
        #plt.title("K-means clustering result:")
        draw_cluster(None, box_cen_new, cluster_all)
        result_file = output_dir+'/'+'k_means_%s.png' % str(box_cen_new[0].cls)
        plt.savefig(result_file); print("k-means result fig saved to: {}".format(os.path.abspath(result_file)))
        plt.show()
        if 0:# plot average dst history
            plt.clf()
            plt.plot([i for i in range(len(iou_history))], iou_history, marker="x")
            plt.show()
    return box_cen_new, i+1, cluster_all

def box_to_string(box_cen, net_shape, v2=False):
    result_str = 'anchors = '
    for box in box_cen:
        result_str += '{:d},{:d},  '.format(round(box.w*net_shape[0]), round(box.h*net_shape[1]))
    return result_str[:-3]

if __name__ == '__main__':
    #parser.add_argument('--max', ) # max_iter
    args = parser.parse_args()
    if args.out: output_dir = args.out
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('Output Dir: {}'.format(os.path.abspath(output_dir)))
    train_file = '../../UAV_for_UAVtest/Uavtrain.txt'

    #label_dir = "E:\Datasets\PASCAL_VOC\VOCdevkit\VOC2007\labels"
    #box_all, max_cls_id = Box.get_boxes_dir(label_dir)
    box_all, max_cls_id = Box.get_boxes_txt(train_file, verbose=args.verbose)
    print(len(box_all))
    if args.by_class:
        test = ['car', 'pedestrian']
        #test = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
        print("Classd-dependent k-means for all %d classes:" % (max_cls_id+1))
        cen_list = []
        cluster_list = []
        overall_str='anchors = '
        box_by_cls = group_boxes_by_class(box_all, max_cls_id)
        for i in range(max_cls_id+1):
            print()
            box_cls = box_by_cls[i]
            box_cen, iters, cluster_all = k_means(box_cls, iou_dst, centroid_avg_side, k=args.k, verbose=args.verbose, vis=args.dont_show, max_iter=1000)
            cen_list.append(box_cen); cluster_list.append(cluster_all) #
        canvas('Class dependent k-means clustering result')
        for i in range(max_cls_id+1):
            print('\ncls %d' %i)
            result_str = box_to_string(cen_list[i], net_shape)
            print(result_str)
            overall_str += result_str[10:] + ',  '
            draw_cluster(None, cen_list[i], cluster_list[i], test[i])
            colors = colors[args.k:]
        result_file = output_dir+'/'+'k_means.png'
        plt.legend(markerscale=5)
        plt.savefig(result_file); print("\nk-means result fig saved to: {}".format(os.path.abspath(result_file)))
        plt.show()
        result_str = overall_str[:-3]
    else:
        box_cen, iters, cluster_all = k_means(box_all, iou_dst, centroid_avg_side, k=args.k, verbose=args.verbose, vis=args.dont_show, max_iter=1000)
        result_str = box_to_string(box_cen, (net_shape))
        for box in box_cen:
            #print('({}, {})'.format(box.w, box.h))
            None
    print(result_str)
