import torchvision.transforms.functional as TF
import random
import math

class Transformer:
    """Apply the following transformations to image and its mask : [brightness, contrast, gamma, saturation, gaussian noise, HFlip, Cropping, Rotation, Shearing]."""

    def __init__(self, brightness_range=(1,1), contrast_range=(1,1), gamma_range=(1,1), saturation_range=(1,1), noise=False, hflip=False, crop=False, rotate=0, shear=0):
        self.brightness_min, self.brightness_max = brightness_range
        self.contrast_min, self.contrast_max = contrast_range
        self.gamma_min, self.gamma_max = gamma_range
        self.saturation_min, self.saturation_max = saturation_range
        self.noise = noise
        self.hflip = hflip
        self.crop = crop
        self.rotate = abs(int(rotate))
        self.shear = abs(int(shear))

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["mask"]
        suffix = ""
        
        #brightness
        if (self.brightness_min != 1 or self.brightness_max != 1) and min(self.brightness_min, self.brightness_max) >= 0:
            brightness = random.uniform(self.brightness_min, self.brightness_max)
            img = TF.adjust_brightness(img, brightness)
            suffix+=f"_b{brightness:.2f}"
        
        #constrast
        if (self.contrast_min != 1 or self.contrast_max != 1) and min(self.contrast_min, self.contrast_max) >= 0:
            constrast = random.uniform(self.contrast_min, self.contrast_max)
            img = TF.adjust_contrast(img, constrast)
            suffix+=f"_c{constrast:.2f}"
        
        #gamma
        if (self.gamma_min != 1 or self.gamma_max != 1) and min(self.gamma_min, self.gamma_max) >= 0:
            gamma = random.uniform(self.gamma_min, self.gamma_max)
            img = TF.adjust_gamma(img, gamma)
            suffix+=f"_g{gamma:.2f}"
        
        #saturation
        if (self.saturation_min != 1 or self.saturation_max != 1) and min(self.saturation_min, self.saturation_max) >= 0:
            saturation = random.uniform(self.saturation_min, self.saturation_max)
            img = TF.adjust_saturation(img, saturation)
            suffix+=f"_s{saturation:.2f}"
        
        #gaussian noise
        if self.noise and random.randint(0,3) == 3:
            mean = 0
            std = 1.
            img += torch.randn(img.size())*std+mean
            suffix += "_n"
                
        #hflip
        if self.hflip and random.randint(0,1) == 1:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
            suffix += "_F"
            
        #crop
        if self.crop:
            w = img.shape[2]
            h = img.shape[1]
            ratio = h/w
            newW = random.randint(int(w/3), w)
            newH= int(ratio*newW)
            top = random.randint(0,h-newH)
            left = random.randint(0,w-newW)
            
            img = TF.crop(img,top,left,newH,newW)
            mask = TF.crop(mask,top,left,newH,newW)
            suffix += "_C"
            
        #rotate
        if self.rotate != 0:
            w = img.shape[2]
            h = img.shape[1]
            
            angle = random.uniform(-self.rotate,self.rotate)
            
            img = TF.affine(img, angle, (0,0), 1, 0, 2)
            mask = TF.affine(mask, angle, (0,0), 1, 0, 0)
            
            angle = math.radians(angle)
            
            if angle > 0:
                x1,y1 = 0,h
                x2,y2 = w,0
            else:
                x1,y1 = 0,0
                x2,y2 = w,h
            x3,y3 = (-w/2)*math.cos(angle)-(h-h/2)*math.sin(angle)+w/2, (-w/2)*math.sin(angle)+(h-h/2)*math.cos(angle)+h/2
            x4,y4 = (w-w/2)*math.cos(angle)-(h-h/2)*math.sin(angle)+w/2, (w-w/2)*math.sin(angle)+(h-h/2)*math.cos(angle)+h/2

            ix,iy = __findIntersection__(x1,y1,x2,y2,x3,y3,x4,y4)

            newW = abs(w-2*int(ix))
            ratio = newW/w
            newH = abs(int(ratio*h))
            
            img = TF.center_crop(img,(newH,newW))
            mask = TF.center_crop(mask,(newH,newW))
            suffix += f"_R{math.degrees(angle):.1f}"
                        
        #shear
        if self.shear != 0:
            w = img.shape[2]
            h = img.shape[1]
            
            shear = random.uniform(-self.shear,self.shear)
                        
            img = TF.affine(img, 0, (0,0), 1, shear, 2)
            mask = TF.affine(mask, 0, (0,0), 1, shear, 0)
            
            shear = math.radians(shear)
            
            if shear > 0:
                x1,y1 = 0,h
                x2,y2 = w,0
            else:
                x1,y1 = 0,0
                x2,y2 = w,h
            x3,y3 = math.tan(shear)*(-h/2), 0
            x4,y4 = math.tan(shear)*h/2, h

            ix,iy = __findIntersection__(x1,y1,x2,y2,x3,y3,x4,y4)

            newW = abs(w-2*int(ix))
            ratio = newW/w
            newH = abs(int(ratio*h))
            
            img = TF.center_crop(img,(newH,newW))
            mask = TF.center_crop(mask,(newH,newW))
            suffix += f"_S{math.degrees(shear):.1f}"

        return { 'image': img, 'mask': mask, 'suffix': suffix }

#https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
def __findIntersection__(x1,y1,x2,y2,x3,y3,x4,y4):
    px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
    py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
    return (px, py)
