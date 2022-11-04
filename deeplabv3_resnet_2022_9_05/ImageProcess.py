import os
import skimage as sk
from skimage import io as sk_io
from skimage import transform as sk_transform
from skimage import img_as_ubyte
import skimage.util as sku
from skimage.filters import threshold_otsu

class ImgProcess:
  def __init__(self, train_dir = "data2/dust_images",raw_img_folder = "raw_target" ,target_size = (1024,1024)):
      self.train_dir = train_dir
      self.raw_img_folder = raw_img_folder
      self.raw_dir = os.path.join(self.train_dir, self.raw_img_folder)
      self.target_size = target_size
      self.resized_input_folder = os.path.join(train_dir, "resized_input")
      self.resized_target_folder = os.path.join(train_dir, "resized_target")

  def Resize(self):

      for root, folders, filenames in os.walk(self.raw_dir):
          if (len(filenames) > 0):
              print(len(filenames), " images detected")
          else:
              print("No images detected in ", self.raw_dir)


      if os.path.exists(self.resized_input_folder):
          print("resize input folder exist")
      else:
          os.makedirs(self.resized_input_folder)

      if os.path.exists(self.resized_target_folder):
          print("target input folder exist")
      else:
          os.makedirs(self.resized_target_folder)

      i = 0

      for image_name in filenames:
          print("Image Name = {}".format(image_name))
          sk_image = sk_io.imread(os.path.join(self.raw_dir, image_name))
          sk_image_resize = sk_transform.resize(sk_image, self.target_size)

          sk_image_resize = sku.img_as_ubyte(sk_image_resize)

          try:
            imgtype = sk_image_resize.shape[2]
          except:
            imgtype = 0

          if imgtype == 0:
            sk_gray_image_resize = sk_image_resize
          elif imgtype == 3:
            sk_gray_image_resize = sk.color.rgb2gray(sk_image_resize)
          elif imgtype == 4:
            sk_image_resize = sk.color.rgba2rgb(sk_image_resize)
            sk_gray_image_resize = sk.color.rgb2gray(sk_image_resize)

          thresh = threshold_otsu(sk_gray_image_resize)
          sk_binary_image_resize = sk_gray_image_resize > thresh

          resized_input_file_path = os.path.join(self.resized_input_folder, str(i)+'.jpg')
          resized_target_file_path = os.path.join(self.resized_target_folder, str(i)+'.jpg')

          sk_io.imsave(fname=resized_input_file_path, arr=img_as_ubyte(sk_binary_image_resize))
          sk_io.imsave(fname=resized_target_file_path, arr=img_as_ubyte(sk_image_resize))

          print(" Processed and Saved")
          i += 1
          progress = (i/len(filenames))*100
          print('%.3f' % progress,"% completed")




if __name__ == "__main__":
    train_dir = 'data/dataset_7k/'
    img_folder = "mask_clean"
    p = ImgProcess(train_dir,img_folder,target_size = (1024,1024))
    p.Resize()

