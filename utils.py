import functools
from torch import nn
import torch
from datetime import datetime
import json
import cv2

def dice_coefficient(input, target):
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return (2. * intersection) / (iflat.sum() + tflat.sum())

def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                m.weight.data.normal_(0.0, gain)
            elif init_type == 'xavier':
                m.weight.data.xavier_normal_(gain=gain)
            elif init_type == 'kaiming':
                m.weight.data.kaiming_normal_(a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                m.weight.data.orthogonal_(gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.fill_(0.0)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, gain)
            m.bias.data.fill_(0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02):
    assert(torch.cuda.is_available())
    net = net.cuda()
    init_weights(net, init_type, gain=init_gain)
    return net

def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(unicode(json.dumps(data, sort_keys=True)))
    log.write(unicode('\n'))
    log.flush()


def extract_frames_from_video(video_path, output_prefix, interval=1, limit=1000):
    cap = cv2.VideoCapture()
    cap.open(video_path)

    counter = 0
    while (cap.isOpened() and counter <= interval * limit):
        ret, frame = cap.read()
        if (counter % interval == 0):
            if (frame == None or frame.shape[0] == 0 or frame.shape[1] == 0):
                break
            size = frame.shape
            frame = frame[:, int((size[1] - size[0]) / 2.0) : int((size[1] - size[0]) / 2.0) + size[0], :]
            size = frame.shape
            print(size)
            cv2.imwrite(output_prefix + "/color_" + str(counter) + ".png", frame)
            # cv2.imshow("windows", frame)
            # cv2.waitKey(100)
        counter += 1
        print(counter)