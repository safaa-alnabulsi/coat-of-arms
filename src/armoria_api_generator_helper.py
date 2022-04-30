import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from src.caption import Caption
from src.armoria_api import ArmoriaAPIPayload, ArmoriaAPIWrapper

class ArmoriaAPIGeneratorHelper:
    def __init__(self, caption_file, folder_name, permutations, start_index=1):
        self.caption_file = caption_file
        self.folder_name = folder_name
        self.permutations = permutations
        self.start_index = start_index

    def generate_caption_file(self):
        for i in range(0, len(self.permutations)):

            label = self.permutations[i]
            text_label = ' '.join(label).strip()
            sample_name = 'image_' + str(i)

            self._write_line_to_file(
                self.caption_file, sample_name + '.png,' + text_label)
    
    def generate_cropped_caption_file_out(self, images_names):
        for image_name, label in zip(images_names, self.permutations):
            text_label = label.strip()
            c = Caption(text_label, support_plural=True)
            if c.is_valid_in_armoria:
                line = image_name + ',' + text_label
                self._write_line_to_file(self.caption_file, line)

    def generate_cropped_caption_file_out_valid(self):
        for label in self.permutations:
            text_label = label.strip()
            c = Caption(text_label, support_plural=True)
            if c.is_valid_in_armoria:
                line = text_label + '.jpg,' + text_label
                self._write_line_to_file(self.caption_file, line)
            
                
    def generate_dataset(self):
#         with open(self.caption_file, 'r', buffering=100000) as f:
        with open(self.caption_file, 'r') as f:
        
            garbage=[next(f) for i in range(self.start_index)]   

            for line in f:
                # skip title
                if 'image,caption' in line:
                    continue

                sample_name, text_label = line.split(',')
                text_label = text_label.strip()
                payload = {}

                try:
                    struc_label = Caption(
                        text_label, support_plural=True).get_structured()
                    payload = ArmoriaAPIPayload(
                        struc_label).get_armoria_payload()
                    api = ArmoriaAPIWrapper(
                        size=500, format="png", coa=payload)

                    image_full_path = self.folder_name + '/images/' + sample_name 

                    self.ensure_dir(image_full_path)

                    api.save_image(image_full_path)

                    print('Image "{}" for label "{}" has been generated succfully' .format(
                        image_full_path, text_label))
                except Exception as e:
                    raise(e)

    def ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def creat_caption_file(self, filename,columns='image,caption'):
        f = open(filename, "w+")
        f.write(columns)
        f.write('\n')
        f.close()

    def _write_line_to_file(self, filename, line):
        with open(filename, 'a') as f:
            f.write(line)
            f.write('\n')
        f.close()

    def _calc_img_pixels(self, image_full_path):
        my_image = Path(image_full_path)
        if not my_image.exists():
            print(f'skipping image {image_full_path}, as it does not exist')

        img     = Image.open(image_full_path).convert("RGB")
        trans   = T.ToTensor()
        img     = trans(img)
        psum    = img.sum()
        psum_sq = (img ** 2).sum()

        return psum, psum_sq

    def add_pixels_column(self, root_folder, new_caption_file,old_caption_file, start_index=1):
        with open(old_caption_file, 'r') as f:
            garbage=[next(f) for i in range(0,start_index+1)]  
            for line in f:
                if 'image,caption' in line:
                    continue
                try:
                    image_name, text_label = line.strip().split(',')
                except ValueError as e:
                    print(f'Invalid label: "{line}" is skipped')
                    continue
                    
                try:
                    image_full_path = root_folder + '/images/' + image_name
                    psum, psum_sq = self._calc_img_pixels(image_full_path)
                except FileNotFoundError as e:
                    print(f'FileNotFoundError: "{image_full_path}"')
                    continue

                newline = f'{image_name},{text_label},{psum},{psum_sq}'
                self._write_line_to_file(new_caption_file, newline)

        f.close()
