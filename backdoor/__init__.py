from backdoor import BadNets, Blended, WaNet
from backdoor.StegaStamp import load_model, encoder_image
from config import opt
import time

e = time.time()
model_path = 'E:/project/project4/DCMH-master/backdoor/ckpt/encoder_imagenet/' #set
model, sees = load_model(model_path)
s = time.time()
# print(s-e)

def select(img, crop_size, trigger_type):

    if 'BadNets' in trigger_type:
        return BadNets.BadNets(img,crop_size=crop_size,model=opt.backdoor_model)

    if 'Blended' in trigger_type:
        return Blended.Blended(img,crop_size=crop_size,model=opt.backdoor_model)

    if 'WaNet' in trigger_type:
        return WaNet.WaNet(img,Height=crop_size)

    if 'StegaStamp' in trigger_type:
        # model_path = './ckpt/encoder_imagenet/'
        # model, sees = load_model(model_path)
        return encoder_image(model, sees, img)