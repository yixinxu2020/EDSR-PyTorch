import os
from PIL import Image


img_name = os.listdir('./training_hr_images')
for i in range(len(img_name)):
    image = img_name[i]
    path = './training_hr_images/' + image
    img = Image.open(path)
    w, h = img.size
    #img_ = img.resize((int(w/3), int(h/3)), Image.BICUBIC)
    img_ = img.resize((int(w/2), int(h/2)), Image.BICUBIC)
    num = str(i + 1)
    save_path = '/home/div/cv/hw4/DIV2K/DIV2K_train_LR_bicubic/X2/' + num.zfill(4) + 'x2.png'
    img_.save(save_path)

