from PIL import Image
import os
#将图片copy到另一文件夹
def copy_image(raw_img_path,new_img_dir):
    os.system(f"cp {raw_img_path} {new_img_dir}")
#图片resize
def resize_image(img_path,new_size=(224,224)):
    img = Image.open(img_path)
    resized_img = img.resize(new_size)
    new_path = img_path[:-4] + 'resize.jpg'
    resized_img.save(new_path)
    resized_img.close()
