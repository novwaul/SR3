### Python Lib.
import os
import sys
### PyTorch Lib.
import torch
import torch.nn as nn
### Multi-GPU Lib.
import torch.multiprocessing as mp
import torch.distributed as dist
### Custom Lib.
from train import DiffTrainer

if __name__ == '__main__': 
    ### Model Lib.
    models = dict()
    for name in os.listdir():
        if os.path.isdir(name) and not name.startswith('.') and not name.startswith('_'):
            module = __import__(name+'.model', fromlist=[name])
            models[name] = module
    
    ### Train and Test Lib.
    trainer = DiffTrainer()
    settings = dict()
    
    """
    [ Hyper-parameters & image paths ]
    Please change the parameters according to your needs.
    """
    ###########################################################################
    settings['steps']=2000 # diffusion training steps
    settings['sample_steps']=100 # diffusion sample steps
    settings['iters']=500000 # training iterations
    settings['lr']=1e-5 # training learning rate
    settings['train_batch_size']=4 # training batch size
    settings['eval_batch_size']=4 # test batch size
    settings['workers']=4 # number of dataloader workers
    settings['report_img_idx'] = [0, 10, 20, 30] # validation image index
    settings['report_img_per'] = 10 # validation image report period
    settings['crop_size'] = 64
    settings['div2k_train_lr_path']=''
    settings['div2k_train_hr_path']=''
    settings['div2k_test_lr_path']=''
    settings['div2k_test_hr_path']=''
    settings['flickr2k_train_lr_path']=''
    settings['flickr2k_train_hr_path']=''
    settings['flickr2k_test_lr_path']=''
    settings['flickr2k_test_hr_path']=''
    ###########################################################################
    """
     [ Hyper-parameters & image paths ]
     End
    """
    
    ### Basic settings
    settings['point_path']='/pnt'
    settings['log_path']='/log'
    settings['states_path']='/states'
    settings['model']=None
    settings['args']=()
    settings['mgpu']=False
    settings['port']=None
    settings['addr']=None
    settings['user_set_devices']=None
    
    ### User settings
    resume = '-r' in sys.argv
    check_param = '-s' in sys.argv
    test = '-t' in sys.argv
    
    if '-v' in sys.argv:
        idx = sys.argv.index('-v')
        settings['point_path'] += '_' + sys.argv[idx+1]
        settings['log_path'] += '_' + sys.argv[idx+1]
        settings['states_path'] += '_' + sys.argv[idx+1]
    
    if '-mgpu' in sys.argv:
        devices = []
        idx = sys.argv.index('-mgpu') + 1
        while idx < len(sys.argv) and '-' not in sys.argv[idx]:
            d = int(sys.argv[idx])
            devices.append(d)
            idx += 1
        # DDP settings
        settings['mgpu']=True
        settings['port']=8888
        settings['addr']='127.0.0.1'
        settings['user_set_devices'] = sorted(devices) if len(device) > 0 else None
    
    if '-m' in sys.argv:
        idx = sys.argv.index('-m')
        model_name = sys.argv[idx+1]
        settings['model'] = getattr(models[model_name], model_name)
        # update paths
        settings['log_path'] = os.getcwd() + '/' + model_name + settings['log_path']
        settings['point_path'] = os.getcwd() + '/' + model_name + settings['point_path']
        settings['states_path'] = os.getcwd() + '/' + model_name + settings['states_path']

    if '-a' in sys.argv:
        idx = sys.argv.index('-a') + 1
        args = []
        while idx < len(sys.argv) and '-' not in sys.argv[idx]:
            a = int(sys.argv[idx])
            args.append(a)
            idx += 1
        settings['args'] = (*args,)
    
    if '-l' in sys.argv:
        idx = sys.argv.index('-l')
        settings['lr'] = float(sys.argv[idx+1])

    if '-p' in sys.argv:
        if settings['mgpu']:
            idx = sys.argv.index('-p')
            settings['port'] = int(sys.argv[idx+1])
        else:
            raise Exception('error: cannot set a port as multi-gpu is not turned on')
    
    if '-i' in sys.argv:
        idx = sys.argv.index('-i')
        settings['iters'] = int(sys.argv[idx+1])

    ### define model
    if settings['model'] == None:
        print('error: please designate a model')
        quit()

    ### check parameter num and exit
    if check_param:
        net = settings['model']() if settings['args'] == None else settings['model'](*settings['args'])
        print(sum(p.numel() for p in net.parameters() if p.requires_grad))
        quit()
    
    ### make log directory
    if not os.path.exists(settings['log_path']):
        os.makedirs(settings['log_path'])

    ### make point directory
    if not os.path.exists(settings['point_path']):
        os.makedirs(settings['point_path'])

    ### make activation directory
    if not os.path.exists(settings['states_path']):
        os.makedirs(settings['states_path'])

    ### execution
    if settings['mgpu'] == True:
        # get device counts
        by_system = torch.cuda.device_count()
        by_user = len(settings['user_set_devices']) if settings['user_set_devices'] != None else by_system
        ngpus_per_node = min(by_user, by_system)
        
        # train
        if test == False:
            mp.spawn(trainer.setup_and_train, nprocs=ngpus_per_node, args=(ngpus_per_node, settings, resume))
        
        # test
        mp.spawn(trainer.setup_and_test, nprocs=ngpus_per_node, args=(ngpus_per_node, settings, resume))
    else:
        # train
        if test == False:
            trainer.setup_and_train(0, ngpus_per_node=1, settings=settings, resume=resume)

        # test
        trainer.setup_and_test(0, ngpus_per_node=1, settings=settings, resume=resume)
