import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets,Synapse_dataset
from tensorboardX import SummaryWriter 
# from engine import *
from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config
from models.MEW10_10 import MEWUNet
from models.unet import UNet
from models.transunet import VisionTransformer
from models.swinunet import SwinUnet
from models.vmunet import VMUNet
from models.mambaunet import MambaUnet
import warnings
from torchvision import transforms
from datasets.dataset import RandomGenerator,myResize
warnings.filterwarnings("ignore")

def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):    
        os.makedirs(outputs)
    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')
    log_config_info(config, logger)
    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    # train_dataset = Synapse_dataset(config.data_path, config, train=True)
    # val_dataset = Synapse_dataset(config.volume_path, config, train=False)
    train_dataset = Synapse_dataset(base_dir=config.data_path, list_dir=config.list_dir, split="train",
                            transform=transforms.Compose(
                                [RandomGenerator(output_size=[config.input_size_h, config.input_size_w])]))
    val_dataset = Synapse_dataset(base_dir=config.volume_path, list_dir=config.list_dir, split="test_vol")

    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)











    print('#----------Prepareing Model----------#')
                        
    #else: raise Exception('network in not right!')
    # model=MEWUNet()
    #model=VisionTransformer(num_classes=1)
    # model=SwinUnet(num_classes=1)
    #model=MambaUnet(img_size=256, num_classes=1)
    model=UNet(3,9)
    #model =VMUNet()
    device = torch.device('cuda:0')
    model = model.to(device)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    #读取断点
    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)



    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()
        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        loss = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )
        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch
           
           # torch.save(mod   el.state_dict(), os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')        )
        logger_info="min_loss:{:.5f}       min_epch:{}".format(min_loss,min_epoch)   
        print(logger_info)
        logger.info(logger_info)
        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')) 

    print("_________________________________")
    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + '/checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(  
                val_loader,
                model,
                criterion,
                logger,
                config,
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )      
if __name__ == '__main__':
    config = setting_config
    main(config)