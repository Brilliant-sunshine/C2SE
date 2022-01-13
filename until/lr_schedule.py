def inv_lr_scheduler(optimizer,iter_num, gamma, power, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
     https://blog.csdn.net/bc521bc/article/details/85864555"""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)

    # i = 0
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr * param_lr[i]
    #     i += 1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

schedule_dict = {"inv": inv_lr_scheduler}