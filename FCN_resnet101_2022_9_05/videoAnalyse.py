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
    video_path = "video/inputs/2022_02_23_clip01.mp4"
    #video_path = "input3.mp4"

    #checkpoint_path = 'unet_7kdata_300epch/Model_7k.pth'
    #checkpoint_path = 'unet_500data_300epch/Model_gen.pth'
    #checkpoint_path = 'logs/FCN/FCN_resnet101/dust/train/20220907-201608/Checkpoints/Model_897_t1.pth'
    checkpoint_path = 'logs/FCN/FCN_resnet101/dust/train/20220907-192159/Checkpoints/Model_100_t1.pth'
    """ Load the model """
    #model = Unet()
    model = M.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=1, aux_loss=None)
    model.cuda()
    model.load_state_dict(torch.load(checkpoint_path))

    """ Reading frames """
    vs = cv2.VideoCapture(video_path)
    _, frame = vs.read()
    H, W, _ = frame.shape
    vs.release()

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('output_2022_02_23_TestTest.avi', fourcc, 10, (W, H), True)

    cap = cv2.VideoCapture(video_path)
    idx = 0
    img_transform = transforms.Compose([
        #transforms.Resize((512,512)),
        transforms.Resize((256,256)),
        #transforms.Resize((1024,1024)),
        transforms.ToTensor()
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dust_array =[]

    frame_array = []
    while True:
        ret, frame = cap.read()
        if ret == False:
            cap.release()
            out.release()
            break

        font = cv2.FONT_HERSHEY_SIMPLEX
        H, W, _ = frame.shape
        ori_frame = frame


        instFrame = ori_frame.copy()
        frame_array.append(instFrame)

        pastFrameMemory = 3

        '''
        if idx == 0:
            first_frame = ori_frame

        derivative_frame = ori_frame
 
        
        print('derivative_frame_id={}, ori_frame_id={}, first_frame_id={}'.format(id(derivative_frame), id(ori_frame), id(first_frame)))
        derivative_frame[(derivative_frame  - first_frame) < 175] = 0
        derivative_frame[(derivative_frame  - first_frame) > 175] = 255
        derivative_frame[derivative_frame[:,:,0] < 175] = 0
        derivative_frame[derivative_frame[:,:,1] < 175] = 0
        derivative_frame[derivative_frame[:,:,2] < 175] = 0
        '''
        print(np.shape(frame_array[-1]))

        try:
            finalFrame = instFrame  - frame_array[-pastFrameMemory]
        except:
            finalFrame = instFrame
        
        #instFrame[(instFrame  - frame_0) > 175] = 0
            

        print("progress = {}".format(idx))

        combine_frame = finalFrame
        
        '''
        ori_frame1 = Image.fromarray(ori_frame)
        #frame = cv2.resize(frame, (256, 256))
        frame = img_transform(ori_frame1)
        #frame = np.expand_dims(frame, axis=0)
        #frame = frame / 255.0
        input = frame.unsqueeze(0).float()
        input = input.cuda()
        mask = torch.sigmoid(model(input)['out']) #[0]

        
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
        invmask2 = (derivative_frame.copy())/255
        print(invmask2[:,:,0])


        #combine_frame[:,:,1] = combine_frame[:,:,1] * invmask[:,:,0]
        
        #combine_frame[:,:,2] = combine_frame[:,:,2] *  invmask2[:,:,0]
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
            'Dust % = {:.1f}'.format(dustRatio), 
            (50, 150), 
            font, 4, 
            (255, 0, 255),
            10, 
            cv2.LINE_4)
        
        model_data = ['Model = fcn_resnet_101_100', 'Size = 4.7GB' ,'Epoch = 200','Train = 807' ,'Validation = 90','val dice coeff = 0.9440','train_platform = RTX3080 16GB']
        
        i = 0
        for data in model_data:
            cv2.putText(combine_frame, 
                data,
                (50, 250 +i), 
                font, 1, 
                (255, 255, 0),
                4, 
                cv2.LINE_4)
            i+= 50

        #cv2.imwrite(f"video/{idx}.png", combine_frame)
        
        '''
        idx += 1

        out.write(combine_frame)
    
    cap.release()