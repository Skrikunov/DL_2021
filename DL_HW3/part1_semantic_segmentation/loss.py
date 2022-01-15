import torch



def calc_val_data(preds, masks, num_classes):
    preds = torch.argmax(preds, dim=1)
    
    # intersection = None # TODO: calc intersection for each class
    # union = None # TODO: calc union for each class
    # target = None # TODO: calc number of pixels in groundtruth mask per class
    # Output shapes: B x num_classes

    n_images = 1
    intersection = torch.zeros([n_images,num_classes])
    union = torch.zeros([n_images,num_classes])
    target = torch.zeros([n_images,num_classes])
    for i in range(n_images):
        for j in range(num_classes):
            intersection[i,j] = ((masks == j)&(preds[i] == j)).sum()
            union[i,j] = ((masks == j)|(preds[i] == j)).sum()
            target[i,j] = (masks == j).sum()

    assert isinstance(intersection, torch.Tensor), 'Output should be a tensor'
    assert isinstance(union, torch.Tensor), 'Output should be a tensor'
    assert isinstance(target, torch.Tensor), 'Output should be a tensor'

    assert intersection.shape == union.shape == target.shape, 'Wrong output shape'
    assert union.shape[0] == masks.shape[0] and union.shape[1] == num_classes, 'Wrong output shape'

    return intersection, union, target

def calc_val_loss(intersection, union, target, eps = 1e-7):
    # mean_iou = None # TODO: calc mean class iou
    # mean_class_rec = None # TODO: calc mean class recall
    # mean_acc = None # TODO: calc mean accuracy

    mean_iou = (intersection/(union+eps)).mean().item()
    mean_class_rec = (intersection/(target+eps)).mean().item()
    mean_acc = (intersection.sum()/(target.sum()+eps)).item()

    return mean_iou, mean_class_rec, mean_acc