import os
from src.caption import Caption
from src.armoria_api import ArmoriaAPIPayload, ArmoriaAPIWrapper

class ArmoriaAPIGeneratorHelper:
    def __init__(self, caption_file, folder_name, permutations):
        self.caption_file = caption_file
        self.folder_name = folder_name
        self.permutations = permutations

    def generate_caption_file(self):
        for i in range(0, len(self.permutations)):

            label = self.permutations[i]
            text_label = ' '.join(label).strip()
            sample_name = 'image_' + str(i)

            self.write_image_label_to_file(
                self.caption_file, sample_name + '.png,' + text_label)

    def generate_dataset(self):

        with open(self.caption_file, 'r', buffering=100000) as f:
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

                    image_full_path = self.folder_name + '/images/' + sample_name + '.png'

                    self.ensure_dir(image_full_path)

                    # api.save_image(image_full_path)

                    print('Image "{}" for label "{}" has been generated succfully' .format(
                        image_full_path, text_label))
                except Exception as e:
                    raise(e)

    def ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def creat_caption_file(self):
        f = open(self.caption_file, "w+")
        f.write('image,caption')
        f.write('\n')
        f.close()

    def write_image_label_to_file(self, filename, line):
        with open(filename, 'a') as f:
            f.write(line)
            f.write('\n')
        f.close()
