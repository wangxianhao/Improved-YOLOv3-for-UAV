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

class Box(object):
    def __init__(self, cls_id, x, y, w, h, c=0):
        self.cls = cls_id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c = c # prediction confidence, set to 0 if it's gt box instead of predicted box

    def __lt__(self, other):
        return self.w*self.h < other.w*other.h

    def __eq__(self, other):
        return self.w==other.w and self.h==other.h

    def __repr__(self):
        return ' '

    def __str__(self):
        message = 'cls_id: {}\n\
                   x: {}\n\
                   y: {}\n\
                   w: {}\n\
                   h: {}\n\
                   confidence: {}\n'.format(self.cls, self.x, self.y, self.w, self.h, self.c)
        return message

    def set_position(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def box_norm(self, im_size=(1242, 375)):
        """
        Normalize kitti format box (x1, y1, x2, y2) to yolo format (x, y, w, h)
        """
        x1, y1, x2, y2 = self.x, self.y, self.w, self.h
        self.x = (x2 + x1) / 2.0 / im_size[0]
        self.y = (y2 + y1) / 2.0 / im_size[1]
        self.w = (x2 - x1) / im_size[0]
        self.h = (y2 - y1) / im_size[1]
        if self.w == 1.0: self.w = .999999
        if self.h == 1.0: self.h = .999999

    def box_restore(self, im_size=(1242, 375)):
        """
        Restore yolo format box (x, y, w, h) to kitti format (x1, y1, x2, y2)
        Inverse operation of box_norm
        """
        x, y, w, h = self.x, self.y, self.w, self.h
        self.x = (x - w / 2.) * im_size[0]
        self.y = (y - h / 2.) * im_size[1]
        self.w = (x + w / 2.) * im_size[0]
        self.h = (y + h / 2.) * im_size[1]

    @classmethod
    def parse_from_label_file(cls, label_path, im_size=(1242, 375), names=None):
        label_stream = open(label_path)
        box_list = []
        for label in label_stream.readlines():
            try:
                box = Box.parse_kitti_label(label, names)
                if im_size != (1242, 375):
                    box.box_norm()
                    box.box_restore(im_size)
            except ValueError:
                box = Box.parse_coco_label(label)
                box.box_restore(im_size)
            box_list.append(box)
        return box_list

    @classmethod
    def parse_kitti_label(cls, label, names=None):
        label = label.strip().split(' ')
        if len(label) == 16:
            cls_name, _, _, _, x1, y1, x2, y2, _, _, _, _, _, _, _, prob = label
        else:
            cls_name, _, _, _, x1, y1, x2, y2, _, _, _, _, _, _, _ = label
            prob = 0
        cls_id = names.index(cls_name.lower()) if names else None
        return Box(cls_id, float(x1), float(y1), float(x2), float(y2), float(prob))

    @classmethod
    def parse_coco_label(cls, label):
        cls_id, x, y, w, h = label.strip().split(' ')
        return Box(int(cls_id), float(x), float(y), float(w), float(h))

    @classmethod
    def get_boxes_dir(cls, label_dir, verbose=False):
        print("Loading label files......")
        label_all = []
        files = [i for i in os.listdir(label_dir) if i.endswith('.txt')]
        print("#img: %d" % len(files))
        for file_name in files:
            label_all.append(os.path.join(label_dir, file_name).strip())
        return cls.get_boxes(label_all, verbose)

    @classmethod
    def get_boxes_txt(cls, train_file, verbose=False):
        """Load boxes from txt list file 'train_file'"""
        print("Loading label files......")
        train_file = open(train_file, newline='\n')
        label_all = []
        bad_list = []
        for l in train_file.readlines():
            l = l.replace('\n', '')
            l = l.replace('.png', '.txt')
            l = l.replace('.jpg', '.txt')
            l = l.replace('JPEGImages', 'labels')
            l = l.replace('/images/', '/labels/') # COCO
            l = l.replace('image_2/', 'labels/') # KITTI
            if not os.path.isfile(l):
                bad_list.append(l)
                continue
            label_all.append(l)
        train_file.close()
        if (len(bad_list)) and not verbose:
            print("Find #%d imgs without label file, set 'verbose' to see details" % len(bad_list))
        if (len(bad_list)) and verbose:
            print("Bad samples without label file:")
            for bad in bad_list:
                print('Cannot open file: %s' % bad)
        print("#img: %d" % len(label_all))
        return cls.get_boxes(label_all, verbose)

    @classmethod
    def get_boxes(cls, label_all, verbose=False):
        box_all = []
        bad_box = []
        max_cls_id = -1
        for label_file in label_all:
             in_stream = open(label_file, newline='\n')
             for l in in_stream.readlines():
                 cls_id, x, y, w, h = l.strip().split(' ')
                 if not(float(w) > 0 and float(h) > 0):
                     bad_box.append(label_file + ' ' + l)
                     continue
                 if(int(cls_id) > max_cls_id): max_cls_id = int(cls_id)
                 box_all.append(Box(int(cls_id), float(x), float(y), float(w), float(h)))
             in_stream.close()
        if len(bad_box) and not verbose:
            print("Find #%d bbox with 'h' or 'w' <= 0, set 'verbose' to see details" % len(bad_box))
        if len(bad_box) and verbose:
            print("Bad boxes with 'h' or 'w' <= 0:")
            for bad in bad_box:
                print(bad.strip())
        print("#bounding box: %d" % len(box_all))
        print("#classes: %d" % (max_cls_id+1))
        return box_all, max_cls_id

    @classmethod
    def filter_boxes(cls, box_list, idx_list):
        box_list_new = []
        for box in box_list:
            cls_id = box.cls
            try:
                box.cls = idx_list.index(cls_id)
                box_list_new.append(box)
            except ValueError:
                pass
        return box_list_new

if __name__ == '__main__':
    train_file = '../../../UAV_for_UAVtest/Uavtrain.txt'
    box_all, max_cls_id = Box.get_boxes_txt(train_file, True)
    #label_dir = 'E:/Datasets/KITTI/training/labels'
    #box_all, max_cls_id = Box.get_boxes_dir(label_dir)
