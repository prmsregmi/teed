import os
import time
import toml
import platform
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from eval import call_ods_ois
import copy
from multiprocessing import Process
# from thop import profile

from types import SimpleNamespace
from .dataset import DATASET_NAMES, BipedDataset, TestDataset, dataset_info
from .loss2 import cats_loss, bdcn_loss2
from .ted import TED # TEED architecture

from .utils.img_processing import save_image_batch_to_disk, visualize_result, count_parameters


os.environ['CUDA_LAUNCH_BLOCKING']="0"
IS_LINUX = True if platform.system()=="Linux" else False

def train_one_epoch(epoch, dataloader, model, criterions, optimizer, device,
                    log_interval_vis, tb_writer, args=None):

    imgs_res_folder = os.path.join(args.output_dir, 'current_res')
    os.makedirs(imgs_res_folder,exist_ok=True)
    show_log = args.show_log
    if isinstance(criterions, list):
        criterion1, criterion2 = criterions
    else:
        criterion1 = criterions

    # Put model in training mode
    model.train()

    l_weight0 = [1.1,0.7,1.1,1.3] # for bdcn loss2-B4
    l_weight = [[0.05, 2.], [0.05, 2.], [0.01, 1.],
                [0.01, 3.]]  # for cats loss [0.01, 4.]
    loss_avg =[]
    for batch_id, sample_batched in enumerate(dataloader):
        images = sample_batched['images'].to(device)  # BxCxHxW
        labels = sample_batched['labels'].to(device)  # BxHxW
        preds_list = model(images)
        loss1 = sum([criterion2(preds, labels,l_w) for preds, l_w in zip(preds_list[:-1],l_weight0)]) # bdcn_loss2 [1,2,3] TEED
        loss2 = criterion1(preds_list[-1], labels, l_weight[-1], device) # cats_loss [dfuse] TEED
        tLoss = loss2+loss1 # TEED

        optimizer.zero_grad()
        tLoss.backward()
        optimizer.step()
        loss_avg.append(tLoss.item())
        if epoch==0 and (batch_id==100 and tb_writer is not None):
            tmp_loss = np.array(loss_avg).mean()
            tb_writer.add_scalar('loss', tmp_loss,epoch)

        if batch_id % (show_log) == 0:
            print(time.ctime(), 'Epoch: {0} Sample {1}/{2} Loss: {3}'
                  .format(epoch, batch_id, len(dataloader), format(tLoss.item(),'.4f')))
        if batch_id % log_interval_vis == 0:
            res_data = []

            img = images.cpu().numpy()
            res_data.append(img[2])

            ed_gt = labels.cpu().numpy()
            res_data.append(ed_gt[2])

            # tmp_pred = tmp_preds[2,...]
            for i in range(len(preds_list)):
                tmp = preds_list[i]
                tmp = tmp[2]
                # print(tmp.shape)
                tmp = torch.sigmoid(tmp).unsqueeze(dim=0)
                tmp = tmp.cpu().detach().numpy()
                res_data.append(tmp)

            vis_imgs = visualize_result(res_data, arg=args)
            del tmp, res_data

            vis_imgs = cv2.resize(vis_imgs,
                                  (int(vis_imgs.shape[1]*0.8), int(vis_imgs.shape[0]*0.8)))
            img_test = 'Epoch: {0} Iter: {1}/{2} Loss: {3}' \
                .format(epoch, batch_id, len(dataloader), round(tLoss.item(),4))

            BLACK = (0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.9
            font_color = BLACK
            font_thickness = 2
            x, y = 30, 30
            vis_imgs = cv2.putText(vis_imgs,
                                   img_test,
                                   (x, y),
                                   font, font_size, font_color, font_thickness, cv2.LINE_AA)
            # tmp_vis_name = str(batch_id)+'-results.png'
            # cv2.imwrite(os.path.join(imgs_res_folder, tmp_vis_name), vis_imgs)
            cv2.imwrite(os.path.join(imgs_res_folder, 'results.png'), vis_imgs)
            # break
    loss_avg = np.array(loss_avg).mean()
    return loss_avg

def validate_one_epoch(epoch, dataloader, model, device, output_dir, arg=None,test_resize=False):
    # XXX This is not really validation, but testing

    # Put model in eval mode
    model.eval()

    with torch.no_grad():
        for _, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            # labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            preds = model(images,single_test=test_resize)
            # print('pred shape', preds[0].shape)
            save_image_batch_to_disk(preds[-1],
                                     output_dir,
                                     file_names,img_shape=image_shape,
                                     arg=arg)


def test(checkpoint_path, dataloader, model, device, output_dir, args,resize_input=False):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    model.eval()
    # just for the new dataset
    # os.makedirs(os.path.join(output_dir,"healthy"), exist_ok=True)
    # os.makedirs(os.path.join(output_dir,"infected"), exist_ok=True)

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            # if not args.test_data == "CLASSIC":
            labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']


            print(f"{file_names}: {images.shape}")
            end = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            preds = model(images, single_test=resize_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            tmp_duration = time.perf_counter() - end
            total_duration.append(tmp_duration)
            save_image_batch_to_disk(preds,
                                     output_dir, # output_dir
                                     file_names,
                                     image_shape,
                                     arg=args)
            torch.cuda.empty_cache()
    total_duration = np.sum(np.array(total_duration))
    print("******** Testing finished in", args.test_data, "dataset. *****")
    print("FPS: %f.4" % (len(dataloader)/total_duration))
    # print("Time spend in the Dataset: %f.4" % total_duration.sum(), "seconds")

def testPich(checkpoint_path, dataloader, model, device, output_dir, args, resize_input=False):
    # a test model plus the interganged channels
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    model.eval()

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            if not args.test_data == "CLASSIC":
                labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            print(f"input tensor shape: {images.shape}")
            start_time = time.time()
            images2 = images[:, [1, 0, 2], :, :]  #GBR
            # images2 = images[:, [2, 1, 0], :, :] # RGB
            preds = model(images,single_test=resize_input)
            preds2 = model(images2,single_test=resize_input)
            tmp_duration = time.time() - start_time
            total_duration.append(tmp_duration)
            save_image_batch_to_disk([preds,preds2],
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     arg=args, is_inchannel=True)
            torch.cuda.empty_cache()

    total_duration = np.array(total_duration)
    print("******** Testing finished in", args.test_data, "dataset. *****")
    print("Average time per image: %f.4" % total_duration.mean(), "seconds")
    print("Time spend in the Dataset: %f.4" % total_duration.sum(), "seconds")

def parse_config(config_path="config.toml"):
    """Parse the TOML config and return (args, train_inf) with flat attribute-style access."""

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")

    config_data = toml.load(config_path)

    # Dataset-dependent values
    TEST_DATA = DATASET_NAMES[config_data['dataset']['choose_test_data']]
    TRAIN_DATA = DATASET_NAMES[config_data['dataset']['choose_train_data']]
    test_inf = dataset_info(TEST_DATA, is_linux=IS_LINUX)
    train_inf = dataset_info(TRAIN_DATA, is_linux=IS_LINUX)

    # Update sections with computed values
    config_data['dataset']['train_data'] = TRAIN_DATA
    config_data['dataset']['test_data'] = TEST_DATA
    config_data['paths']['input_dir'] = train_inf['data_dir']
    config_data['paths']['input_val_dir'] = test_inf['data_dir']
    config_data['training']['train_list'] = train_inf['train_list']
    config_data['training']['test_list'] = test_inf['test_list']
    config_data['testing']['test_img_width'] = test_inf['img_width']
    config_data['testing']['test_img_height'] = test_inf['img_height']
    config_data['data_mean']['mean_test'] = test_inf['mean']
    config_data['data_mean']['mean_train'] = train_inf['mean']

    # Flatten the configuration to a single dictionary
    flat_config = {}
    for section, values in config_data.items():
        if isinstance(values, dict):
            flat_config.update(values)
        else:
            flat_config[section] = values

    args = SimpleNamespace(**flat_config)
    return args, train_inf

def run_teed(args, train_inf):
    # Tensorboard summary writer
    # torch.autograd.set_detect_anomaly(True)
    tb_writer = None
    training_dir = os.path.join(args.output_dir,args.train_data)
    os.makedirs(training_dir,exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, args.train_data,args.checkpoint_data)
    if args.tensorboard and not args.is_testing:
        # from tensorboardX import SummaryWriter  # previous torch version
        from torch.utils.tensorboard import SummaryWriter # for torch 1.4 or greather
        tb_writer = SummaryWriter(log_dir=training_dir)
        # saving training settings
        training_notes =[args.version_notes+ ' RL= ' + str(args.lr) + ' WD= '
                          + str(args.wd) + ' image size = ' + str(args.img_width)
                          + ' adjust LR=' + str(args.adjust_lr) +' LRs= '
                          + str(args.lrs)+' Loss Function= BDCNloss2 + CAST-loss2.py '
                          + str(time.asctime())+' trained on '+args.train_data]
        info_txt = open(os.path.join(training_dir, 'training_settings.txt'), 'w')
        info_txt.write(str(training_notes))
        info_txt.close()
        print("Training details> ",training_notes)

    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')
    # torch.cuda.set_device(args.use_gpu) # set a desired gpu

    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")
    # print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Trainimage mean: {args.mean_train}')
    print(f'Test image mean: {args.mean_test}')


    # Instantiate model and move it to the computing device
    model = TED().to(device)
    # model = nn.DataParallel(model)
    ini_epoch = 0
    if not args.is_testing:
        if args.resume:
            checkpoint_path2= os.path.join(args.output_dir, 'CLASSIC', args.checkpoint_data)
            ini_epoch=3
            model.load_state_dict(torch.load(checkpoint_path2,
                                         map_location=device))

        # Training dataset loading...
        dataset_train = BipedDataset(args.input_dir,
                                     img_width=args.img_width,
                                     img_height=args.img_height,
                                     train_mode='train',
                                     arg=args
                                     )
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                  num_workers=args.workers)
    # Test dataset loading...
    dataset_val = TestDataset(args.input_val_dir,
                              test_data=args.test_data,
                              img_width=args.test_img_width,
                              img_height=args.test_img_height,
                              test_list=args.test_list, arg=args
                              )
    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.workers)
    # Testing
    if_resize_img = False if args.test_data in ['BIPED', 'CID', 'MDBD'] else True
    if args.is_testing:

        checkpoint_path_for_test = os.path.join(args.output_dir, args.checkpoint_data)
        output_dir = os.path.join(args.res_dir, checkpoint_path_for_test)
        print(f"output_dir: {output_dir}")

        test(checkpoint_path_for_test, dataloader_val, model, device,
             output_dir, args,if_resize_img)

        # Count parameters:
        num_param = count_parameters(model)
        print('-------------------------------------------------------')
        print('TED parameters:')
        print(num_param)
        print('-------------------------------------------------------')
        return

    criterion1 = cats_loss #bdcn_loss2
    criterion2 = bdcn_loss2#cats_loss#f1_accuracy2
    criterion = [criterion1,criterion2]
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.wd)

    # Count parameters:
    num_param = count_parameters(model)
    print('-------------------------------------------------------')
    print('TEED parameters:')
    print(num_param)
    print('-------------------------------------------------------')

    # Main training loop
    seed=1021
    adjust_lr = args.adjust_lr
    k=0
    set_lr = args.lrs#[25e-4, 5e-6]
    eval_processes = []
    for epoch in range(ini_epoch,args.epochs):
        if epoch%5==0: # before 7

            seed = seed+1000
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print("------ Random seed applied-------------")
        # adjust learning rate
        if adjust_lr is not None:
            if epoch in adjust_lr:
                lr2 = set_lr[k]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr2
                k+=1
        # Create output directories

        output_dir_epoch = os.path.join(args.output_dir,args.train_data, str(epoch))
        img_test_dir = os.path.join(output_dir_epoch, args.test_data + '_res')
        os.makedirs(output_dir_epoch,exist_ok=True)
        os.makedirs(img_test_dir,exist_ok=True)

        print("**************** Validating the training from the scratch **********")
        avg_loss =train_one_epoch(epoch,dataloader_train,
                        model, criterion,
                        optimizer,
                        device,
                        args.log_interval_vis,
                        tb_writer=tb_writer,
                        args=args)
        validate_one_epoch(epoch,
                           dataloader_val,
                           model,
                           device,
                           img_test_dir,
                           arg=args, test_resize=if_resize_img)

        # Save model after end of every epoch
        torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                   os.path.join(output_dir_epoch, '{0}_model.pth'.format(epoch)))
        
        # run inference and evaluation here
        
        checkpoint_path_for_test = os.path.join(output_dir_epoch, '{0}_model.pth'.format(epoch))
        output_dir = os.path.join(args.res_dir,checkpoint_path_for_test)
        print(f"checkpoint_path_for_test: {checkpoint_path_for_test}", output_dir)
        tmp_args = copy.copy(args)
        tmp_args.is_testing = True
        test(checkpoint_path_for_test, dataloader_val, model, device, output_dir, tmp_args,if_resize_img)


        if True:
            time.sleep(5)
            # Launch eval process
            eval_proc = Process(target=call_ods_ois, args=("CLASSIC", os.path.join(output_dir, "fused")))
            eval_proc.start()
            eval_processes.append(eval_proc)
            
        if tb_writer is not None:
            tb_writer.add_scalar('loss',
                                 avg_loss,
                                 epoch+1)
        print('Last learning rate> ', optimizer.param_groups[0]['lr'])

    num_param = count_parameters(model)
    print('-------------------------------------------------------')
    print('TEED parameters:')
    print(num_param)
    print('-------------------------------------------------------')

def main(id):
    # os.system(" ".join(command))
    args, train_info = parse_config()
    args.output_dir = args.output_dir + f"checkpoints_{id}"
    run_teed(args, train_info)

if __name__ == '__main__':
    main()
