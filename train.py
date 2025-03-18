import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import wandb
import os
import json
import valid
from utils import utils
from utils import sam
from utils import option
from data import dataset
from model import HTR_VT
from functools import partial

def compute_loss_ce(args, model, image, batch_size, criterion, text):
    preds = model(image, args.mask_ratio, args.max_span_length, use_masking=True)
    preds = preds.float()
    
    # Reshape predictions for cross entropy loss
    # [batch_size, seq_len, vocab_size]
    batch_size, seq_len, vocab_size = preds.size()
    
    # Flatten predictions to [batch_size * seq_len, vocab_size]
    preds_flat = preds.reshape(-1, vocab_size)
    
    # Flatten targets to [batch_size * seq_len]
    targets_flat = text.cuda().reshape(-1)
    
    # Apply cross entropy loss
    loss = criterion(preds_flat, targets_flat)
    
    return loss


def main():

    args = option.get_args_parser()
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    
    # Initialize wandb if enabled, otherwise use TensorBoard
    if args.use_wandb:
        
        wandb.init(project="custom_htr", 
                  name=args.exp_name,
                  config=vars(args))
        logger.info("Using Weights & Biases for logging")
    else:
        writer = SummaryWriter(args.save_dir)
        logger.info("Using TensorBoard for logging")

    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1], max_seq_length=args.max_seq_length)

    total_param = sum(p.numel() for p in model.parameters())
    logger.info('total_param is {}'.format(total_param))

    model.train()
    model = model.cuda()
    model_ema = utils.ModelEma(model, args.ema_decay)
    model.zero_grad()

    logger.info('Loading train loader...')
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_bs,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers,
                                               collate_fn=partial(dataset.SameTrCollate, args=args))
    train_iter = dataset.cycle_data(train_loader)

    logger.info('Loading val loader...')
    val_dataset = dataset.myLoadDS(args.val_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_bs,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers)

    optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=1e-7, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    cross_entropy_converter = utils.CrossEntropyConverter(args.max_seq_length)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=cross_entropy_converter.pad_token)

    best_cer, best_wer = 1e+6, 1e+6
    train_loss = 0.0

    #### ---- train & eval ---- ####

    # Initialize early stopping variables
    early_stop_counter = 0
    early_stop_best_val = float('inf')  # Track best validation metric (CER)
    
    for nb_iter in range(1, args.total_iter):

        optimizer, current_lr = utils.update_lr_cos(nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

        optimizer.zero_grad()
        batch = next(train_iter)
        image = batch[0].cuda()
        text_ce = cross_entropy_converter.encode(batch[1])
        batch_size = image.size(0)
        # loss = compute_loss(args, model, image, batch_size, criterion, text, length)
        # loss.backward()
        loss_ce = compute_loss_ce(args, model, image, batch_size, criterion, text_ce)
        loss_ce.backward()
        optimizer.first_step(zero_grad=True)
        loss_ce = compute_loss_ce(args, model, image, batch_size, criterion, text_ce)
        loss_ce.backward()
        optimizer.second_step(zero_grad=True)
        model.zero_grad()
        model_ema.update(model, num_updates=nb_iter / 2)
        train_loss += loss_ce.item()

        if nb_iter % args.print_iter == 0:
            train_loss_avg = train_loss / args.print_iter

            logger.info(f'Iter : {nb_iter} \t LR : {current_lr:0.5f} \t training loss : {train_loss_avg:0.5f} \t ' )

            if args.use_wandb:
                wandb.log({
                    'train/lr': current_lr,
                    'train/loss': train_loss_avg
                }, step=nb_iter)
            else:
                writer.add_scalar('./Train/lr', current_lr, nb_iter)
                writer.add_scalar('./Train/train_loss', train_loss_avg, nb_iter)
            train_loss = 0.0

        if nb_iter % args.eval_iter == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_cer, val_wer, preds, labels = valid.validation(model_ema.ema,
                                                                             criterion,
                                                                             val_loader,
                                                                             cross_entropy_converter)

                if val_cer < best_cer:
                    logger.info(f'CER improved from {best_cer:.4f} to {val_cer:.4f}!!!')
                    best_cer = val_cer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, 'best_CER.pth'))
                    
                    # Reset early stopping counter when CER improves
                    early_stop_counter = 0
                    early_stop_best_val = val_cer
                else:
                    # Increment early stopping counter when no improvement
                    early_stop_counter += 1
                    logger.info(f'Early stopping counter: {early_stop_counter}/{args.early_stop_patience}')

                if val_wer < best_wer:
                    logger.info(f'WER improved from {best_wer:.4f} to {val_wer:.4f}!!!')
                    best_wer = val_wer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, 'best_WER.pth'))

                correct_predictions = sum(p == l for p, l in zip(preds, labels))
                total_predictions = len(labels)
                accuracy = correct_predictions / total_predictions

                logger.info(
                    f'Val. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} \t Accuracy: {correct_predictions}/{total_predictions} ({accuracy:0.4f})')
                
                

                if args.use_wandb:
                    wandb.log({
                        'val/loss': val_loss,
                        'val/CER': val_cer,
                        'val/WER': val_wer,
                        'val/Accuracy': accuracy,
                        'val/best_CER': best_cer,
                        'val/best_WER': best_wer
                    }, step=nb_iter)
                else:
                    writer.add_scalar('./VAL/CER', val_cer, nb_iter)
                    writer.add_scalar('./VAL/WER', val_wer, nb_iter)
                    writer.add_scalar('./VAL/bestCER', best_cer, nb_iter)
                    writer.add_scalar('./VAL/bestWER', best_wer, nb_iter)
                    writer.add_scalar('./VAL/val_loss', val_loss, nb_iter)
                    writer.add_scalar('./VAL/Accuracy', accuracy, nb_iter)
                # Check if early stopping criteria is met
                if args.use_early_stopping and early_stop_counter >= args.early_stop_patience:
                    logger.info(f'Early stopping triggered after {nb_iter} iterations. No improvement for {args.early_stop_patience} evaluations.')
                    break
                    
                model.train()


if __name__ == '__main__':
    main()