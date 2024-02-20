import torchvision
import os
import json


root_dir = '/tmp/'
train_dataset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True)
eval_dataset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True)


idx = 0
def build_image_text_dataset(root, folder, dataset):
    results = []
    global idx
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    if not os.path.exists(os.path.join(root,folder)):
        os.mkdir(os.path.join(root,folder))
    for img, label_idx in dataset:
        item  = {}
        imgname = "IMG{:06d}.png".format(idx)
        filename = os.path.join(root, folder, imgname)
        idx = idx + 1
        caption = 'this is a picture of {}.'.format(labels[label_idx])
        img.save(filename)
        item['caption_id'] = idx
        item['image_id'] = idx
        item['caption'] = caption
        item['image_path'] = filename
        results.append(item)
    return results

def gen_caption_meta(root, name, meta):
    save_path = os.path.join(root, name+'.json')
    with open(save_path, 'w') as fw:
        fw.write(json.dumps(meta, indent=4))

train_results = build_image_text_dataset(root_dir, 'train', train_dataset)
gen_caption_meta(root_dir, 'train', train_results)

eval_results = build_image_text_dataset(root_dir, 'eval', eval_dataset)
gen_caption_meta(root_dir, 'eval', eval_results)