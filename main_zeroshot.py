import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import *
from torch.autograd import Variable
import numpy as np
import json

from sklearn.covariance import LedoitWolf, OAS, GraphicalLassoCV, GraphicalLasso
from tqdm import tqdm

torch.cuda.set_device(4)

_tokenizer = _Tokenizer()
train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])


def get_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    parser.add_argument('--config', default='configs/zero_shot/caltech101.yaml', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()
    return args        


def pp_estimate_eigen(logits, labels):
    probs = F.softmax(logits, dim=-1)
    _, K = probs.size()
    P = torch.zeros(K, K)
    for k in range(K):
        indices = (labels == k).nonzero(as_tuple=True)[0]
        class_probs = probs[indices]
        class_average = class_probs.mean(dim=0) if len(indices) > 0 else torch.zeros(K)
        P[k] = class_average

    P = P.t()
    pp = power_iteration(P)
    # print("=> after estimating P_p")
    # logits -= torch.log(pp + 1e-12).to(logits.device)
    # acc = cls_acc(logits, labels)
    # print(f"* accuracy: {acc:.2f}%")
    return pp, logits

def compute_confidence(logits, T):
    scale_logits = logits / T
    probability = F.softmax(scale_logits, dim=-1)
    confidence, _ = torch.max(probability, dim=-1)
    confidence = confidence.mean().item()

    return confidence

def search_T(logits, tao, zero_conf):
    conf = compute_confidence(logits, tao)

    if abs(zero_conf  - conf) / zero_conf < 0.1:
        return tao

    if zero_conf >= conf :
        tao /= 1.1
    elif zero_conf < conf:
        tao *= 1.1
    return search_T(logits, tao, zero_conf) 

def power_iteration(A, tol=1e-3, max_iter=100):
    n = A.shape[0]
    x = torch.full((n,), 1/n, dtype=A.dtype, device=A.device)   

    for _ in range(max_iter):
        x_next = torch.mv(A, x)
        x_next = x_next / torch.norm(x_next, p=1)  
        x_next = torch.clamp(x_next, min=0)  

        if torch.norm(x_next - x, p=1) < tol:
            break

        x = x_next

    return x

def run(cfg, clip_weights, clip_model, llm_fea):  
    
    # Parameter Estimation.
    with torch.no_grad():      
        cls_num = clip_weights.shape[-1]
        fea_dim = clip_weights.shape[0]
        prmt_num = llm_fea.shape[1]
        test_bz = test_features.shape[0]
        # val_bz = val_features.shape[0]


        clip_weights_ave = llm_fea.mean(dim=1)
        llm_weights_ave = clip_weights_ave / clip_weights_ave.norm(dim=-1, keepdim=True)
        
        clip_logits = 100. * test_features @ clip_weights
        llm_ave_logits = 100. * test_features @ llm_weights_ave.T
        clip_acc = cls_acc(clip_logits, test_labels)
        llm_acc1 = cls_acc(llm_ave_logits, test_labels)

        llm_weights = llm_fea.reshape(-1, fea_dim)
        llm_weights /= llm_weights.norm(dim=-1, keepdim=True)
        llm_logits_ave = 100. * test_features @ llm_weights.T
        llm_logits_ave = llm_logits_ave.reshape(test_bz, cls_num, prmt_num).softmax(dim=1)
        llm_logits_ave = llm_logits_ave.mean(dim=-1, keepdim=False)
        llm_acc2 = cls_acc(llm_logits_ave, test_labels)

        print('clip_zero:', clip_acc)
        print('llm_acc1', llm_acc1)
        print('llm_acc2', llm_acc2)

        lamda = 0.9
        used_samples = torch.cat((test_features, llm_weights.reshape(-1, fea_dim)), dim=0)
        cov_inv = torch.linalg.inv((1 - lamda) * used_samples.T.cov() * torch.eye(fea_dim).cuda() + lamda * torch.eye(fea_dim).cuda())
        x_part = (test_features @ cov_inv @ test_features.T ).diag() / 2
        mu_part = (llm_weights_ave @ cov_inv @ llm_weights_ave.T).diag() / 2
        crs_part = llm_weights_ave @ cov_inv @ test_features.T
        lda_logits = (crs_part - mu_part[:,None] - x_part[None,:]).T
        print('lda logits', cls_acc(lda_logits, test_labels))  

        InMap_logits = (test_features @ image_opt(test_features, clip_weights, sinkhorn(clip_logits / 100, gamma=1), test_labels, alpha=0.6))
        InMap_logits_v1 = (test_features @ image_opt(test_features, clip_weights, sinkhorn(llm_ave_logits / 100, gamma=1), test_labels, alpha=0.6))
        InMap_logits_v2 = (test_features @ image_opt(test_features, llm_weights_ave.T, sinkhorn(llm_ave_logits / 100, gamma=1), test_labels, alpha=0.6))
        InMap_logits_v3 = (test_features @ image_opt(test_features, clip_weights, sinkhorn(lda_logits , gamma=1), test_labels, alpha=0.6))
        InMap_logits_v4 = (test_features @ image_opt(test_features, llm_weights_ave.T, sinkhorn(lda_logits , gamma=1), test_labels, alpha=0.6))

        print('InMap', cls_acc(InMap_logits, test_labels))
        print('InMap_template_llm-logits',cls_acc(InMap_logits_v1, test_labels))
        print('InMap_llm-template_llm-logits',cls_acc(InMap_logits_v2, test_labels))
        print('InMap_template_lda-logits',cls_acc(InMap_logits_v3, test_labels))
        print('InMap_template_lda-logits',cls_acc(InMap_logits_v4, test_labels))

        lda_logits_sinkhorn = sinkhorn(lda_logits)
        llm_ave_logits_sinkhorn = sinkhorn(llm_ave_logits)

        ensemble_logits = 100 * InMap_logits_v3 + 100 * InMap_logits_v1
        cmp_logits = ensemble_logits.clone().detach()
        used_debias_logits = ensemble_logits.clone().detach()
        used_debias_labels = used_debias_logits.argmax(dim=-1)
        n = 0
        pp_orig = torch.zeros(cls_num)
        while n <= 10:
            values, _ = used_debias_logits.topk(2, dim=-1)
            diff = torch.abs(values[:,0] - values[:,1])
            epsilon = 1 * 1 / cls_num
            index = torch.where(diff > epsilon)[0]
            used_debias_logits = used_debias_logits[index]
            used_debias_labels = used_debias_logits.argmax(dim=-1)
            pp, _ = pp_estimate_eigen(used_debias_logits, used_debias_labels)
            ensemble_logits = ensemble_logits - torch.log(pp + 1e-12).cuda()
            print('debias acc', n, cls_acc(ensemble_logits, test_labels), torch.norm(pp - pp_orig, p=1))
            used_debias_logits = ensemble_logits.clone()
            used_debias_labels = used_debias_logits.argmax(dim=-1)
            pp_orig = pp
            n += 1


        # upper bound
        pp, _ = pp_estimate_eigen(cmp_logits, test_labels)
        upper_output = cmp_logits - torch.log(pp + 1e-12).cuda()
        print('unpper bound acc', cls_acc(upper_output, test_labels))

    print('test')

def main():
    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # Load cfg for conditional prompt.
    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
        
    notune_accs = {"1": [], "2": [], "3": [], "3407":[]}
    
    for seed in [1]:
    # for seed in [3407]:
        cfg["seed"] = seed
        random.seed(seed)
        torch.manual_seed(seed) 
    
        print("Preparing dataset.")
        global train_loader_F
        global test_features, test_labels
        global val_features, val_labels
        if cfg['dataset'] != "imagenet":
            dataset = build_dataset(cfg, cfg['dataset'], cfg['root_path'], cfg['shots']) 

            with open(cfg['gpt3_prompt_file']) as f:
                gpt3_prompt = json.load(f)
                gpt_fea_dct = gpt_llm_fea(dataset.classnames, gpt3_prompt, clip_model.float(), dataset.template)
            with open(cfg['prompt_cafo']) as f:
                cafo_prompt = json.load(f)
                cafo_fea_dct = gpt_llm_fea(dataset.classnames, cafo_prompt, clip_model.float(), dataset.template)

            merged_dict = defaultdict(list)
            merged_dict = {key: torch.cat([gpt_fea_dct[key], cafo_fea_dct[key]]) for key in gpt_fea_dct}
            
            del gpt_fea_dct
            del cafo_fea_dct
            print('test')

            test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False)

            test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

        clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model.float())   
        llm_fea = torch.stack([merged_dict[name.replace('_',' ')] for name in dataset.classnames])


        notune_acc = run(cfg, clip_weights, clip_model, llm_fea)
        notune_accs[str(cfg["seed"])].append(notune_acc)
    
if __name__ == '__main__':
    main()
