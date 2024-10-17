import pandas as pd
import torchvision.transforms as transforms
import math
import clip
from models.simulator import simulator
import torch
#编码方案1 随机裁剪
def random_crop(img,crop_size):
    random_cropor = transforms.RandomCrop(crop_size)
    crop_img = random_cropor(img)
    return crop_img
# 均聚物约束
def homopolymer(sequence, max_homopolymer):
    if max_homopolymer > len(sequence):
        return True

    missing_segments = ["A" * (1 + max_homopolymer), "C" * (1 + max_homopolymer), "G" * (1 + max_homopolymer),
                        "T" * (1 + max_homopolymer)]

    for missing_segment in missing_segments:
        if missing_segment in "".join(sequence):
            return False
    return True
# cg含量约束
def cg_content(motif, max_content):
    """
    Check the C and G content of requested DNA sequence.

    :param motif: requested DNA sequence.
    :param max_content: maximum content of C and G, which means GC content is in [1 - max_content, max_content].

    :return: whether the DNA sequence can be considered as valid for DNA synthesis and sequencing.
    """
    return (1 - max_content) <= float(motif.count("C") + motif.count("G")) / float(len(motif)) <= max_content
# 相似度约束
def similarity(trans_seq,raw_seq,threshold=0.95):
    seq_pair = pd.DataFrame({
        "a":raw_seq,
        "b":trans_seq
    })
    yields = simulator(seq_pair)
    if yields>=threshold:
        return True
    else:
        return False

def check(trans_seq,raw_seq, max_homopolymer=math.inf, max_content=1, min_free_energy=None,max_ata_count=0,threshold=0.95):
    if not homopolymer(trans_seq, max_homopolymer):
        return False
    if not cg_content(trans_seq, max_content):
        return False
    if not similarity(trans_seq,raw_seq,threshold):
        return False

    return True

device = 'cpu'
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def adjust_coding(img,raw_seq,count_sum,clip_model,preprocess,encoder,crop_size=224,device='cpu'):
    count = 0
    while count < count_sum:
        trans_img = random_crop(img,crop_size)
        pre_img = preprocess(trans_img).unsqueeze(0).to(device)
        pre_img_feature = clip_model.encode_image(pre_img)
        encoder.eval()
        with torch.no_grad():
            trans_seq = encoder.feature_to_seq(pre_img_feature)
        if check(trans_seq,raw_seq):
            return trans_seq
        else:
            count += 1
    return False

















