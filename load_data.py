import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.data_aug import ColorAugmentation
import os
from dataset.coco import COCODetectionDataset

def return_sampler(train_dset, test_dset, world_size, rank):
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dset, num_replicas=world_size, rank=rank
    )
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dset, num_replicas=world_size, rank=rank
    )
    
    return train_sampler, test_sampler


def get_dataloader(train_dset, test_dset, train_sampler, test_sampler, args, collate_fn=None):
    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, collate_fn=collate_fn)
    
    return train_loader, test_loader



def load_datasets(args, only_val=True):
    import os
    download=False
    
    def load_cifar10(path, img_size=224):
        ratio = 224.0 / float(img_size)
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        train_dataset = datasets.CIFAR10(
            path,
            transform=transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
            download=download
        )
        test_dataset = datasets.CIFAR10(
            path,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
            download=download,
            train=False
        )
        
        
        train_sampler = None
        test_sampler = None
        
        if args.distributed:
            train_sampler, test_sampler = return_sampler(train_dataset, test_dataset, args.world_size, args.local_rank)

        train_loader, test_loader = get_dataloader(train_dataset, test_dataset, train_sampler, test_sampler, args)
        
        return train_loader, test_loader
    
    
    def load_cifar100(path, img_size=224):
        ratio = 224.0 / float(img_size)
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        train_dataset = datasets.CIFAR100(
            path,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
            download=download
        )
        test_dataset = datasets.CIFAR100(
            path,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
            download=download,
            train=False
        )

        train_sampler = None
        test_sampler = None

        if args.distributed:
            train_sampler, test_sampler = return_sampler(train_dataset, test_dataset, args.world_size, args.local_rank)
            
        train_loader, test_loader = get_dataloader(train_dataset, test_dataset, train_sampler, test_sampler, args)
        
        return train_loader, test_loader
        
        
    def load_imagenet1k(path, img_size=224, only_val=True):
        ratio = 224.0 / float(img_size)
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        traindir = os.path.join(path, 'ILSVRC2012_img_train')
        valdir = os.path.join(path, 'ILSVRC2012_img_val')
        
        train_sampler = None
        val_sampler = None
        
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ColorAugmentation(),
                normalize,
            ]))
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(int(256 * ratio)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
        ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=(train_sampler is None), sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=val_sampler)
        
        if only_val:
            return None, val_loader
        else:
            return train_loader, val_loader
        
        
    def load_coco_det(root_path):
        ann_dir = os.path.join(root_path, 'annotations')
        train_dir = os.path.join(root_path, 'train2017')
        val_dir = os.path.join(root_path, 'val2017')
        test_dir = os.path.join(root_path, 'test2017')
        
        train_ann = os.path.join(ann_dir, 'instances_train2017.json')
        val_ann = os.path.join(ann_dir, 'instances_val2017.json')
        test_ann = os.path.join(ann_dir, 'image_info_test2017.json')

        # train_dataset = COCODataset(train_dir, train_ann)
        # val_dataset = COCODataset(val_dir, val_ann)
        
        train_trans, val_trans = coco_det_transforms()
        
        train_sampler, val_sampler = None, None
        
        train_dataset = COCODetectionDataset(train_dir, train_ann, transforms=train_trans)
        val_dataset = COCODetectionDataset(val_dir, val_ann, transforms=val_trans)
        
        # train_dataset = datasets.coco.CocoDetection(train_dir, train_ann, transforms=train_trans)
        # val_dataset = datasets.coco.CocoDetection(val_dir, val_ann, transforms=val_trans)
        
        train_loader, val_loader = get_dataloader(train_dataset, val_dataset, train_sampler, val_sampler, args, collate_fn)
        
        return train_loader, val_loader
            
        
    
    if args.dataset == 'cifar10':
        return load_cifar10('/root/data/pytorch_datasets')
    
    elif args.dataset == 'cifar100':
        return load_cifar100('/root/data/pytorch_datasets')
    
    elif args.dataset == 'imagenet1k':
        return load_imagenet1k(path='/root/data/img_type_datasets/ImageNet-1K', only_val=only_val)
    
    elif args.dataset == 'coco_det':
        return load_coco_det('/root/data/mmdataset/coco')
    
    
def coco_det_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    ])
    
    return train_transforms, test_transforms


def collate_fn(batch):
    return tuple(zip(*batch))