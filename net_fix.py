import torch
import os


def load_and_fix_checkpoint(args, model, opt):
    checkpoint = torch.load(args.resume, map_location=f'cuda')
    
    model.load_state_dict(checkpoint['state_dict'])
    if args.num_cls is not None and args.num_cls != 1000:
        model.module.change_last_layer(args.num_cls)
        
    opt.load_state_dict(checkpoint['optimizer'])
    for p in opt.param_groups:
        p['lr'] = args.base_lr
    
        
    return model, opt


# def change_last_layer(model, num_cls):
#     model.module.change_last_layer(num_cls)
    
#     return model
    