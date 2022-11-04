import os

from torchvision.transforms import transforms
from PIL import Image
from model.Unet import Unet
import torchvision.models as M

import numpy as np
import cv2
import torch

if __name__ == "__main__":
    """ Video Path """

    video_flder = "video/inputs/"
    #video_name = "2022_02_24_clip01.mp4"
    #video_name = "Tobruk Rd.mov"
    video_name = "Video_multiple views.mp4"  
    
    video_path = video_flder + video_name

    Model_name = "Unet"

    #Dataset = "dataset_100"
    Dataset = "dataset_897"

    output_name = "output_" + Model_name + "_" + Dataset + "_" + video_name


    #checkpoint_path = 'unet_7kdata_300epch/Model_7k.pth'
    #checkpoint_path = 'unet_500data_300epch/Model_gen.pth'

    #checkpoint_path = 'logs/UNet/VanillaUnet/dust/train/20221021-150046/Checkpoints/Model_100_t1.pth' #model trained for 100
    checkpoint_path = 'logs/UNet/VanillaUnet/dust/train/20221022-201337/Checkpoints/Model_897_t1.pth' #model trained for 897




    """ Load the model """
    model = Unet()
    model.cuda()
    model.load_state_dict(torch.load(checkpoint_path))

    """ Reading frames """
    vs = cv2.VideoCapture(video_path)
    _, frame = vs.read()
    H, W, _ = frame.shape
    vs.release()

    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out = cv2.VideoWriter(output_name, fourcc, 24, (W, H), True)

    cap = cv2.VideoCapture(video_path)
    idx = 0
    img_transform = transforms.Compose([
        #transforms.Resize((512,512)),
        transforms.Resize((256,256)),
        #transforms.Resize((1024,1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dust_array =[]

    while True:
        ret, frame = cap.read()
        if ret == False:
            cap.release()
            out.release()
            break

        font = cv2.FONT_HERSHEY_SIMPLEX
        H, W, _ = frame.shape
        ori_frame = frame
        ori_frame1 = Image.fromarray(ori_frame)
        #frame = cv2.resize(frame, (256, 256))
        frame = img_transform(ori_frame1)
        #frame = np.expand_dims(frame, axis=0)
        #frame = frame / 255.0
        input = frame.unsqueeze(0).float()
        input = input.cuda()
        #mask = torch.sigmoid(model(input)['out']) #[0]
        mask = torch.sigmoid(model(input)) #[0] #unet only

        
        print(mask.shape)
        mask = mask.squeeze(0).squeeze(0)
        
        
        #mask = mask > 0.99


        mask = mask.detach().cpu().numpy()       
        mask = mask.astype(np.float32)
        mask = cv2.resize(mask, (W, H))
        mask = np.expand_dims(mask, axis=-1)
        #mask = mask > 0.9
        #mask = mask > 0.1
        dust_array.append(np.sum(mask))

        print(ori_frame.shape)
        print(mask.shape)
        invmask = mask.copy()
        invmask = abs(mask - 1)
        combine_frame = ori_frame

        combine_frame[:,:,1] = combine_frame[:,:,1] * invmask[:,:,0]
        #combine_frame[:,:,0] = np.add(combine_frame[:,:,0], mask[:,:,0]*250)
        print(combine_frame.shape)
        combine_frame = combine_frame.astype(np.uint8)
        total_pixels = (combine_frame.shape[0] * combine_frame.shape[1])

        if len(dust_array) > 30:
            dust_pixel = (np.sum(dust_array[-31:-1]))/30
        else:
            dust_pixel = (np.sum(dust_array))/len(dust_array)

        print('total = {:.1f} dust = {:.1f}'.format(total_pixels,dust_pixel))
        dustRatio = (dust_pixel/total_pixels)*100
        #dustRatio = dust_pixel

        cv2.putText(combine_frame, 
            'Dust Pixel Ratio% = {:.1f}'.format(dustRatio), 
            (50, 150), 
            font, 1, 
            (255, 0, 255),
            4, 
            cv2.LINE_4)
        
        #model_data = ['Model = Unet', 'Epochs = 200','Train = 807' ,'Validation = 90','val dice coeff = 0.9335','train_platform = RTX3080 16GB']
        model_data = ['Model = {}'.format(Model_name), Dataset, 'train_platform = RTX3080 16GB']
        #model_data = ['Model = Unet', 'dataset_897','train_platform = RTX3080 16GB']
        
        i = 0
        for data in model_data:
            cv2.putText(combine_frame, 
                data,
                (50, 250 +i), 
                font, 1, 
                (255, 0, 255),
                4, 
                cv2.LINE_4)
            i+= 50

        #cv2.imwrite(f"video/{idx}.png", combine_frame)
        idx += 1

        out.write(combine_frame)


    
    cap.release()