import datetime

import torch
from torch import nn
import sys

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modules.HistoryLog import LossHistory
from src import models
from src import ctc
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *


####################################################################
#
# Construct the model and the CTC module (which may not be needed)
#
####################################################################

def get_CTC_module(hyp_params):
    a2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_a, out_seq_len=hyp_params.l_len)
    v2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_v, out_seq_len=hyp_params.l_len)
    return a2l_module, v2l_module

def initiate(hyp_params, train_loader, valid_loader, test_loader, loss_history):
    # getattr() 函数用于返回一个对象属性值。
    model = getattr(models, hyp_params.model+'Model')(hyp_params)


    if hyp_params.use_cuda:
        model = model.cuda()

    # summary(model)
    # from torchviz import make_dot
    # text = torch.randn(4, 20, 300).requires_grad_(True)  # 定义一个网络的输入值
    # audio = torch.randn(4, 400, 74).requires_grad_(True)  # 定义一个网络的输入值
    # vision = torch.randn(4, 500, 35).requires_grad_(True)  # 定义一个网络的输入值
    #
    # out = model(text, audio, vision)  # 获取网络的预测值
    # # graph = make_dot(out, params=dict(model.named_parameters()))
    # # graph.render(filename='cnn', view=False, format='png')
    # graph = make_dot(out, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    # graph.render(filename='cnn1', view=False, format='png')
    # graph = make_dot(out)
    # graph.render(filename='cnn2', view=False, format='png')
    # graph = make_dot(out, dict(list(model.named_parameters()) + [('text', text), ('audio',audio),('vision', vision)]))
    # graph.render(filename='cnn3', view=False, format='png')


    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    # 对齐了的时候是不需要ctc了的，MULT模型是适用于对齐或者不对齐情况的，它不需要额外的ctc
    # if hyp_params.aligned or hyp_params.model=='MULT':
    #     # 对齐需要语音流和视觉流的对齐
    #     ctc_criterion = None
    #     ctc_a2l_module, ctc_v2l_module = None, None
    #     ctc_a2l_optimizer, ctc_v2l_optimizer = None, None
    # else:
    #     from warpctc_pytorch import CTCLoss
    #     ctc_criterion = CTCLoss()
    #     ctc_a2l_module, ctc_v2l_module = get_CTC_module(hyp_params)
    #     if hyp_params.use_cuda:
    #         ctc_a2l_module, ctc_v2l_module = ctc_a2l_module.cuda(), ctc_v2l_module.cuda()
    #     ctc_a2l_optimizer = getattr(optim, hyp_params.optim)(ctc_a2l_module.parameters(), lr=hyp_params.lr)
    #     ctc_v2l_optimizer = getattr(optim, hyp_params.optim)(ctc_v2l_module.parameters(), lr=hyp_params.lr)

    ctc_criterion = None
    ctc_a2l_module, ctc_v2l_module = None, None
    ctc_a2l_optimizer, ctc_v2l_optimizer = None, None
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)

    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'ctc_a2l_module': ctc_a2l_module,
                'ctc_v2l_module': ctc_v2l_module,
                'ctc_a2l_optimizer': ctc_a2l_optimizer,
                'ctc_v2l_optimizer': ctc_v2l_optimizer,
                'ctc_criterion': ctc_criterion,
                'scheduler': scheduler,
                # 'writer': writer,
                'loss_history': loss_history
                }
    # loss_history = LossHistory(hyp_params.logs_dir, model, hyp_params.dataset)
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']    
    
    ctc_a2l_module = settings['ctc_a2l_module']
    ctc_v2l_module = settings['ctc_v2l_module']
    ctc_a2l_optimizer = settings['ctc_a2l_optimizer']
    ctc_v2l_optimizer = settings['ctc_v2l_optimizer']
    ctc_criterion = settings['ctc_criterion']
    
    scheduler = settings['scheduler']

    # writer = settings['writer']
    loss_history = settings['loss_history']
    

    def train(model, optimizer, criterion, ctc_a2l_module, ctc_v2l_module, ctc_a2l_optimizer, ctc_v2l_optimizer, ctc_criterion, epoch):
        epoch_loss = 0
        model.train()
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        loop = tqdm(train_loader, total=int(hyp_params.n_train/hyp_params.batch_size), desc=f'Epoch Train [{epoch}/{hyp_params.num_epochs}]')
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loop):
            # iemocap no_aligned: batch_Y{4,4}, sample_ind{4}, text{4,20,300}, audio{4,400,74},vision{4,500,35},
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            
            model.zero_grad()
            if ctc_criterion is not None:
                ctc_a2l_module.zero_grad()
                ctc_v2l_module.zero_grad()
                
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.long()
            
            batch_size = text.size(0)
            batch_chunk = hyp_params.batch_chunk
            
            ######## CTC STARTS ######## Do not worry about this if not working on CTC
            if ctc_criterion is not None:
                ctc_a2l_net = nn.DataParallel(ctc_a2l_module) if batch_size > 10 else ctc_a2l_module
                ctc_v2l_net = nn.DataParallel(ctc_v2l_module) if batch_size > 10 else ctc_v2l_module

                audio, a2l_position = ctc_a2l_net(audio) # audio now is the aligned to text
                vision, v2l_position = ctc_v2l_net(vision)
                
                ## Compute the ctc loss
                l_len, a_len, v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
                # Output Labels
                l_position = torch.tensor([i+1 for i in range(l_len)]*batch_size).int().cpu()
                # Specifying each output length
                l_length = torch.tensor([l_len]*batch_size).int().cpu()
                # Specifying each input length
                a_length = torch.tensor([a_len]*batch_size).int().cpu()
                v_length = torch.tensor([v_len]*batch_size).int().cpu()
                
                ctc_a2l_loss = ctc_criterion(a2l_position.transpose(0,1).cpu(), l_position, a_length, l_length)
                ctc_v2l_loss = ctc_criterion(v2l_position.transpose(0,1).cpu(), l_position, v_length, l_length)
                ctc_loss = ctc_a2l_loss + ctc_v2l_loss
                ctc_loss = ctc_loss.cuda() if hyp_params.use_cuda else ctc_loss
            else:
                ctc_loss = 0
            ######## CTC ENDS ########
                
            combined_loss = 0
            net = nn.DataParallel(model) if batch_size > 100 else model
            if batch_chunk > 1:
                raw_loss = combined_loss = 0
                text_chunks = text.chunk(batch_chunk, dim=0)
                audio_chunks = audio.chunk(batch_chunk, dim=0)
                vision_chunks = vision.chunk(batch_chunk, dim=0)
                eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)
                
                for i in range(batch_chunk):
                    text_i, audio_i, vision_i = text_chunks[i], audio_chunks[i], vision_chunks[i]
                    eval_attr_i = eval_attr_chunks[i]
                    preds_i, hiddens_i = net(text_i, audio_i, vision_i)
                    
                    if hyp_params.dataset == 'iemocap':
                        preds_i = preds_i.view(-1, 2)
                        eval_attr_i = eval_attr_i.view(-1)
                    raw_loss_i = criterion(preds_i, eval_attr_i) / batch_chunk
                    raw_loss += raw_loss_i
                    raw_loss_i.backward()
                ctc_loss.backward()
                combined_loss = raw_loss + ctc_loss
            else:
                preds, hiddens = net(text, audio, vision)
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                raw_loss = criterion(preds, eval_attr)
                combined_loss = raw_loss + ctc_loss
                combined_loss.backward()
            
            if ctc_criterion is not None:
                torch.nn.utils.clip_grad_norm_(ctc_a2l_module.parameters(), hyp_params.clip)
                torch.nn.utils.clip_grad_norm_(ctc_v2l_module.parameters(), hyp_params.clip)
                ctc_a2l_optimizer.step()
                ctc_v2l_optimizer.step()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            optimizer.zero_grad()




            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            # if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
            #     avg_loss = proc_loss / proc_size
            #     elapsed_time = time.time() - start_time
            #     print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
            #           format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
            #     proc_loss, proc_size = 0, 0
            #     start_time = time.time()
            elapsed_time = time.time() - start_time
            loop.set_postfix(**{'Time/Batch(ms)': elapsed_time * 1000,
                                'Train Loss': proc_loss / proc_size})

            loop.update(1)
            start_time = time.time()


        return epoch_loss / hyp_params.n_train

    def evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=False, epoch=0):
        model.eval()
        loader = test_loader if test else valid_loader
        type = "Test" if test else "Evalute"
        total_loss = 0.0
    
        results = []
        truths = []

        with torch.no_grad():
            loop = tqdm(loader, total=int(((hyp_params.n_test if test else hyp_params.n_valid) / hyp_params.batch_size)), desc=f'Epoch {type} [{epoch}/{hyp_params.num_epochs}]')
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loop):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1) # if num of labels is 1
            
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'iemocap':
                            eval_attr = eval_attr.long()
                        
                batch_size = text.size(0)
                
                if (ctc_a2l_module is not None) and (ctc_v2l_module is not None):
                    ctc_a2l_net = nn.DataParallel(ctc_a2l_module) if batch_size > 100 else ctc_a2l_module
                    ctc_v2l_net = nn.DataParallel(ctc_v2l_module) if batch_size > 100 else ctc_v2l_module
                    audio, _ = ctc_a2l_net(audio)     # audio aligned to text
                    vision, _ = ctc_v2l_net(vision)   # vision aligned to text
                
                net = nn.DataParallel(model) if batch_size > 100 else model
                preds, _ = net(text, audio, vision)
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                loss = criterion(preds, eval_attr).item()
                total_loss += loss * batch_size

                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)

                loop.set_postfix(loss=loss)
                loop.update(1)

                
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        # mosei criterion is L1Loss
        epoch_train_loss = train(model, optimizer, criterion, ctc_a2l_module, ctc_v2l_module, ctc_a2l_optimizer, ctc_v2l_optimizer, ctc_criterion, epoch)
        val_loss, _, _ = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=False, epoch=epoch)
        test_loss, _, _ = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=True, epoch=epoch)

        
        end = time.time()
        duration = end-start
        scheduler.step(val_loss)    # Decay learning rate by validation loss

        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        print("-"*50)

        loss_history.append_loss(epoch, epoch_train_loss, val_loss, test_loss)

        
        if val_loss < best_valid:
            print(f"Saved model at {loss_history.model_save_path}/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name, model_path=loss_history.model_save_path)
            best_valid = val_loss

    model = load_model(hyp_params, name=hyp_params.name, model_path=loss_history.model_save_path)
    _, results, truths = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=True)

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True, loss_history.loss_txt)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True, loss_history.loss_txt)
    elif hyp_params.dataset == 'iemocap':
        eval_iemocap(results, truths, loss_txt=loss_history.loss_txt)

    sys.stdout.flush()
    input('[Press Any Key to start another run]')
