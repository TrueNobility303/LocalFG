import os
import torch
from functools import partial
from trainer.utils import AverageMeter, accuracy, set_bn_train, set_bn_eval
from trainer.hooks import set_net_hook, set_hook
from trainer.gradients import save_auxiliary_gradients, save_gradient, set_gradient, set_auxiliary_gradients, \
    get_gradient_computation_function, compute_loss_acc 

from trainer.estimators import get_estimator

from trainer.per_sample_gradients import linear_gradient, conv_gradient, batchnorm_gradient
from trainer.trackers import reset_tracker, print_tracker, log_perf_to_wandb, print_perf
from tqdm import tqdm
from model.utils import init_weights
import matplotlib.pyplot as plt 

# define the util functions
def inner_product(a,b):
    return (a * b).sum()

def compute_loss(net, data, target, criterion):
    pred = net(data)
    loss = criterion(pred, target)
    return loss 

def normalize(a, eps):
    return a / (torch.norm(a) + eps)

def _train_epoch(dataloader, net, criterion, optimizer, target='global', guess='local', space='weight', device=None,
                 args=None):
    
    # Setting target = 'global' means use eng-to-end loss
    compute_target = partial(get_gradient_computation_function(target), dest='target', space=space)

    # Setting guess = 'local' means using Local Guess to guide the perturbation direction
    compute_guess = partial(get_gradient_computation_function(guess), dest='guess', space=space)

    # Random guess: original Forward Gradient Training 
    if args.training.guess == 'random':
        compute_guess = partial(compute_guess, noise_type=args.training.noise_type)

    compute_estimator = get_estimator(space)
    # tracker
    net.train()
    net.to(device)
    criterion.to(device)
    reset_tracker(net)

    # record the loss/acc
    for block in net.blocks:
        block.auxnet.loss = AverageMeter()
        block.auxnet.accs = AverageMeter()
    
    for data, target in tqdm(dataloader, desc= "Training...", position=1):
        data, target = data.to(device), target.to(device)

        if guess == 'ntk':
            for k, block in enumerate(net.blocks):
                if k < len(net.blocks) - 1:
                    block.auxnet.apply(lambda m: init_weights(m, mode=args.model.weight_init))

        optimizer.zero_grad()
        net.apply(set_bn_train)

        # For weight perturbation
        # Set hooks in forward pass: save module.inputs
        # Set hooks in backward pass: module['target'] = linear_gradient(module.inputs, grad_output) 
        handles = set_net_hook(net, space, 'target')

        # Compute the global/local loss
        # This function will trigger the hooks
        compute_target(net, data, target, criterion)

        # set module.weight/bias.saved_grad be the global/local gradients
        save_gradient(net.blocks[-1])
        if args.training.target == 'local':
            save_auxiliary_gradients(net)
        
        for h in handles:
            h.remove()

        optimizer.zero_grad()
        net.apply(set_bn_eval)
        for k, block in enumerate(net.blocks):
            if k < len(net.blocks) - 1:
                block.auxnet.apply(set_bn_train)

        # For weight perturbation
        # Set hooks in backward pass:
        # If use local guess, set  module['guess'] = linear_gradient(module.inputs, grad_output) 
        # If use random guess, just set module['guess'] = random.normal 
                
        handles = set_net_hook(net, space, 'guess')
        compute_guess(net, data, target, criterion)

        if guess == 'local':
            save_auxiliary_gradients(net)
        for h in handles:
            h.remove()

        # module.grad = guess * <target, guess> / <guess, guess>
                   
        compute_estimator(net)
        
        # the last block can use standrad backprop
        set_gradient(net.blocks[-1])

        # other cases: ntk or local target
        if guess in ['fixed-ntk', 'ntk']:
            for k, block in enumerate(net.blocks):
                if k < len(net.blocks) - 1:
                    block.auxnet.zero_grad()
        elif target == 'local' or guess == 'local':
            set_auxiliary_gradients(net)

        optimizer.step()

    # return the gloabl loss stored in the last block
    return net.blocks[-1].auxnet.loss.avg, net.blocks[-1].auxnet.accs.avg

def _train_epoch_with_local_FG_weight(dataloader, net, criterion, optimizer, target='global', guess='local', space='weight', device=None,
                 args=None):

    compute_target = partial(get_gradient_computation_function(target), dest='target', space=space)
    compute_guess = partial(get_gradient_computation_function(guess), dest='guess', space=space)

    # tracker
    net.train()
    net.to(device)
    criterion.to(device)
    reset_tracker(net)

    for block in net.blocks:
        block.auxnet.loss = AverageMeter()
        block.auxnet.accs = AverageMeter()
    

    for data, target in tqdm(dataloader, desc= "Training...", position=1):
        data, target = data.to(device), target.to(device)

        # target gradient
        optimizer.zero_grad()
        net.apply(set_bn_train)
        handles = set_net_hook(net, space, 'target')

        compute_target(net, data, target, criterion)

        save_gradient(net.blocks[-1])
        if args.training.target == 'local':
            save_auxiliary_gradients(net)
        for h in handles:
            h.remove()

        # guess gradient
        optimizer.zero_grad()
        net.apply(set_bn_eval)
        for k, block in enumerate(net.blocks):
            if k < len(net.blocks) - 1:
                block.auxnet.apply(set_bn_train)
        handles = set_net_hook(net, space, 'guess')

        compute_guess(net, data, target, criterion)

        if guess == 'local':
            save_auxiliary_gradients(net)
        for h in handles:
            h.remove()
        
        # compute gradients
        # The following codes differ from https://github.com/streethagore/ForwardLocalGradient/blob/main/trainer/forward_gradient.py

        # Preprocess module 
            # 1. Take average through axis=batch for both target and guess 
            # 2. Normalize module.guess
        
        for k, block in enumerate(net.blocks):
            if k < len(net.blocks) - 1:
                for module in block.block.modules():
                    if isinstance(module, torch.nn.modules.Linear) or \
                            isinstance(module, torch.nn.modules.conv._ConvNd) or \
                            isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        if module.weight.requires_grad:
                            module.target['weight'] = torch.sum(module.target['weight'], dim=0)
                            module.guess['weight'] = torch.sum(module.guess['weight'], dim=0)
                            normalize(module.guess['weight'], args.training.eps)
                            
                        if module.bias is not None and module.bias.requires_grad:
                            module.target['bias'] = torch.sum(module.target['bias'], dim=0)
                            module.guess['bias'] = torch.sum(module.guess['bias'], dim=0)
                            normalize(module.guess['bias'], args.training.eps)
        
        projected_grad = 0
        for k, block in enumerate(net.blocks):
            if k < len(net.blocks) - 1:
                for module in block.block.modules():
                    if isinstance(module, torch.nn.modules.Linear) or \
                            isinstance(module, torch.nn.modules.conv._ConvNd) or \
                            isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        if module.weight.requires_grad:
                            projected_grad += inner_product(module.target['weight'], module.guess['weight'])

                        if module.bias is not None and module.bias.requires_grad:
                            projected_grad += inner_product(module.target['bias'], module.guess['bias'])

        # Now projected_grad = < target, guess >
    
        # Set gradients for optimizer to update 

        for k, block in enumerate(net.blocks):
            if k < len(net.blocks) - 1:
                for module in block.block.modules():
                    if isinstance(module, torch.nn.modules.Linear) or \
                            isinstance(module, torch.nn.modules.conv._ConvNd) or \
                            isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        if module.weight.requires_grad:
                            module.weight.grad =  projected_grad * module.guess['weight']

                        if module.bias is not None and module.bias.requires_grad:
                            module.bias.grad = projected_grad * module.guess['bias']

        # the last block can use standrad backprop
        set_gradient(net.blocks[-1])
        if target == 'local' or guess == 'local':
            set_auxiliary_gradients(net)
        optimizer.step()

    # return the gloabl loss stored in the last block
    return net.blocks[-1].auxnet.loss.avg, net.blocks[-1].auxnet.accs.avg

def _train_epoch_with_local_FG_activity(dataloader, net, criterion, optimizer, target='global', guess='local', space='weight', device=None,
                 args=None):
    compute_target = partial(get_gradient_computation_function(target), dest='target', space=space)
    compute_guess = partial(get_gradient_computation_function(guess), dest='guess', space=space)
    
    # tracker
    net.train()
    net.to(device)
    criterion.to(device)
    reset_tracker(net)
    
    for block in net.blocks:
        block.auxnet.loss = AverageMeter()
        block.auxnet.accs = AverageMeter()
    
    for data, target in tqdm(dataloader, desc= "Training...", position=1):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        net.apply(set_bn_train)

        # record the backprop signal in the hook
        def backward_hook_for_activity_perturbation(module, grad_input, grad_output, dest):
            grad_output = grad_output[0].contiguous()
            setattr(module, dest, grad_output)

        def set_net_hook_for_activity_perturbation(net, dest):
            handles = []
            for k, block in enumerate(net.blocks):
                if k < len(net.blocks) - 1:
                    handles.extend(set_hook(block.block, partial(backward_hook_for_activity_perturbation, dest=dest)))
            return handles

        # Compute the global loss and record the backward signals
        handles = set_net_hook_for_activity_perturbation(net, dest='target')
        compute_target(net, data, target, criterion)

        save_gradient(net.blocks[-1])
        if args.training.target == 'local':
            save_auxiliary_gradients(net)
        
        for h in handles:
            h.remove()

        optimizer.zero_grad()
        net.apply(set_bn_eval)
        for k, block in enumerate(net.blocks):
            if k < len(net.blocks) - 1:
                block.auxnet.apply(set_bn_train)

        # Compute the local loss and record the backward signals        
        handles = set_net_hook_for_activity_perturbation(net, dest='guess')
        compute_guess(net, data, target, criterion)

        if guess == 'local':
            save_auxiliary_gradients(net)
        for h in handles:
            h.remove()

        # Computed the projection
        projected_grad = 0
        for k, block in enumerate(net.blocks):
            if k < len(net.blocks) - 1:
                for module in block.block.modules():
                    if isinstance(module, torch.nn.modules.Linear) or \
                            isinstance(module, torch.nn.modules.conv._ConvNd) or \
                            isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        normalize(module.guess, args.training.eps)
                        if (module.weight.requires_grad) or  (module.bias is not None and module.bias.requires_grad):
                            projected_grad += inner_product(module.target, module.guess)

        # Compuate the estimator of activity perturbation
        for k, block in enumerate(net.blocks):
            if k < len(net.blocks) - 1:
                for module in block.block.modules():
                    if isinstance(module, torch.nn.modules.Linear) or \
                        isinstance(module, torch.nn.modules.conv._ConvNd) or \
                        isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        
                        # get per sample gradient 
                        if isinstance(module, torch.nn.modules.Linear):
                            per_sample_grad = linear_gradient(module.inputs, projected_grad * module.guess)
                        elif isinstance(module, torch.nn.modules.conv._ConvNd):
                            per_sample_grad = conv_gradient(module, module.inputs, projected_grad * module.guess)
                        elif isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                            per_sample_grad = batchnorm_gradient(module, module.inputs, projected_grad * module.guess)

                        # set the gradient to weight or bias, respectively
                        if module.weight.requires_grad:
                            module.weight.grad = per_sample_grad['weight'].sum(dim=0)
                        if module.bias is not None and module.bias.requires_grad:
                            module.bias.grad = per_sample_grad['bias'].sum(dim=0)
                    
        set_gradient(net.blocks[-1])
        if target == 'local' or guess == 'local':
            set_auxiliary_gradients(net)

        optimizer.step()

    # return the gloabl loss stored in the last block
    return net.blocks[-1].auxnet.loss.avg, net.blocks[-1].auxnet.accs.avg

@torch.no_grad()
def validate(dataloader, net, criterion, device, args):
    net.to(device)
    net.eval()
    for block in net.blocks:
        block.auxnet.loss = AverageMeter()
        block.auxnet.accs = AverageMeter()

    for data, target in tqdm(dataloader, desc= "Testing...", position=2):
        data, target = data.to(device), target.to(device)
        x = data
        for block in net.blocks:
            x = block(x)
            pred = block.auxnet(x)
            loss = criterion(pred, target)
            accs = accuracy(pred, target)
            block.auxnet.loss.update(loss.item())
            block.auxnet.accs.update(accs.item())

    return net.blocks[-1].auxnet.loss.avg, net.blocks[-1].auxnet.accs.avg

def get_train_epoch_function(algorithm):
    if algorithm == 'std':
        from trainer.backprop import train_epoch
        return train_epoch
    elif algorithm == 'localonly':
        from trainer.localonly import train_epoch
        return train_epoch
    elif algorithm == 'avg-weight-fast':
        from trainer.fast_avg_weight_pertub import train_epoch
        return train_epoch
    elif algorithm == 'generic':
        return _train_epoch
    elif algorithm == 'LocalFG-W':
        return _train_epoch_with_local_FG_weight
    elif algorithm == 'LocalFG-A':
        return _train_epoch_with_local_FG_activity
    else:
        return None


def train(train_loader, test_loader, net, criterion, optimizer, scheduler, n_epoch, args=None):
    train_epoch = get_train_epoch_function(args.training.algorithm)
    target, guess, space, device = args.training.target, args.training.guess, args.training.space, args.device

    train_loss_lst = []
    train_acc_lst = []
    test_loss_lst = []
    test_acc_lst = []

    epochs = tqdm(range(n_epoch), desc=f"Epoch ... (0/{n_epoch})", position=0)
    for epoch in epochs:
        train_loss, train_acc = train_epoch(train_loader, net, criterion, optimizer, target=target, guess=guess,
                                            space=space, device=device, args=args)

        print_perf(net, epoch, train_loss, train_acc, scheduler.get_last_lr()[0], 'train')
        if args is not None and args.wandb.status:
            log_perf_to_wandb(net, epoch, scheduler, 'train')

        test_loss, test_acc = validate(test_loader, net, criterion, device, args)

        # record the metrics
        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)
        test_loss_lst.append(test_loss)
        test_acc_lst.append(test_acc)

        # save the metrics 
        with open(os.path.join(args.output_dir, "train_loss.txt") , "w") as f:
            print(train_loss_lst, file=f)
        with open(os.path.join(args.output_dir, "test_loss.txt" ) , "w") as f:
            print(test_loss_lst, file=f)
        with open(os.path.join(args.output_dir, "train_acc.txt") , "w") as f:
            print(train_acc_lst, file=f)
        with open(os.path.join(args.output_dir, "test_acc.txt") , "w") as f:
            print(test_acc_lst, file=f)
    
        # plot the metrics per epoch
        epoch_lst = [i for i in range(len(train_loss_lst))]
        plt.figure()
        plt.subplot(2,2,1)
        
        plt.plot(epoch_lst, train_acc_lst)
        plt.ylabel('accuracy')
        plt.xlabel('#epochs')
        plt.title('train')

        plt.subplot(2,2,2)
        plt.plot(epoch_lst, test_acc_lst)
        plt.ylabel('accuracy')
        plt.xlabel('#epochs')
        plt.title('test')

        plt.subplot(2,2,3)
        plt.plot(epoch_lst, train_loss_lst)
        plt.xlabel('#epochs')
        plt.ylabel('loss')
        plt.title('train')

        plt.subplot(2,2,4)
        plt.plot(epoch_lst, test_loss_lst)
        plt.xlabel('#epochs')
        plt.ylabel('loss')
        plt.title('test')

        plt.tight_layout()
        path = os.path.join(args.output_dir, "fig.png")
        plt.savefig(path)
        plt.close()

        print_perf(net, epoch, test_loss, test_acc, scheduler.get_last_lr()[0], 'test')
        if args is not None and args.wandb.status:
            log_perf_to_wandb(net, epoch, scheduler, 'test')

        # if args.histogram_folder is not None and not epoch % 20:
        #     print_tracker(net, space, os.path.join(args.histogram_folder, f'epoch_{epoch}'))

        if args.save_model > 0 and not epoch % args.save_model:
            net.cpu()
            fpath = f'output/{args.model.arch}-{args.aux_loss.auxnet.aux_type}-{args.training.target}-{args.training.guess}-{args.training.space}-epoch_{epoch}.weights'
            torch.save({'args': args, 'weights': net.state_dict()}, fpath)

        scheduler.step()

        epochs.desc = f"Epoch ... {epoch + 1}/{n_epoch}"
