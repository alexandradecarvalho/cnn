class Crowd(data.Dataset):
    def __init__(self, root_path, pad_size, trans, downsample_ratio=8):
        self.root_path = root_path
        self.trans = trans
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        self.annot_list = sorted(glob(os.path.join(self.root_path, '*.mat')))
        self.max_size = pad_size
        print('number of img: {}'.format(len(self.im_list))) 
        print(self.max_size)

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        ann_path = img_path.replace('.jpg', '_ann.mat')
        img = self.trans(self.padding(Image.open(img_path).convert('RGB')))
        keypoints = loadmat(ann_path)['annPoints']
        return img, len(keypoints)

    def padding(self, img):
        img = FF.pad(img, (0, self.max_size - img.size[0]))
        return FF.pad(img, (0, self.max_size - img.size[1]))