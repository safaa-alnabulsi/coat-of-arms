import streamlit as st
import pandas as pd
from pathlib import Path
import json
from PIL import Image
import numpy as np


@st.cache
def load_data(all_or_valid):
    if all_or_valid == "All":
        data_dir = Path("data/cropped_coas/out")
    else:
        data_dir = Path("data/cropped_coas/out_valid")

    images, captions = [], []

    for image_fn in data_dir.iterdir():
        if image_fn.suffix == ".jpg" and not image_fn.name.startswith("."):
            image = np.asarray(Image.open(image_fn))
            images.append(image)
            captions.append(image_fn.stem)

    return images, captions

st.sidebar.markdown("## Filters/Options")
all_or_valid = st.sidebar.radio("Filter", ("Valid", "All"))
show_only_with_generated_image = st.sidebar.checkbox("Show only with generated image", value=True)

images, captions = load_data(all_or_valid)            

from src.label_checker_automata import LabelCheckerAutomata
from src.armoria_api import ArmoriaAPIPayload, ArmoriaAPIWrapper


class Caption:

    # (e.g. "A lion rampant")
    def __init__(self, label, support_plural=False): 
        self.label = label
        self.support_plural = support_plural

    @property
    def is_valid(self):
        simple_automata = LabelCheckerAutomata(support_plural=self.support_plural)
        return simple_automata.is_valid(self.label)

    # (e.g. “com")
    def get_automata_parsed(self): 
        simple_automata = LabelCheckerAutomata(support_plural=self.support_plural)
        return simple_automata.parse_label(self.label)

    # ({“Colors”:[‘A’], “Object/Modifier”: [“lion rampant”]})
    def get_aligned(self):
        simple_automata = LabelCheckerAutomata(support_plural=self.support_plural)
        parsed_label = simple_automata.parse_label(self.label)
        return simple_automata.align_parsed_label(self.label, parsed_label)

    def get_armoria_payload_dict(self):
        return ArmoriaAPIPayload(self.label.split()).get_armoria_payload()

    # (['A', 'lion rampant’])
    def get_tokenized(self):
        pass

    #  [1,3,2,4,2]
    def get_numericalized():
        pass


df = pd.DataFrame.from_dict({
    "image": images,
    "caption": captions,
})


generated_images = []

for image, caption_str in zip(images, captions):
    # st.image(image, caption=caption)
    caption = Caption(caption_str, support_plural=False)
    if caption.is_valid:
        try:
            armoria_payload = caption.get_armoria_payload_dict()
            generated_image = np.asarray(ArmoriaAPIWrapper(
                size=500,
                format="png", 
                coa=armoria_payload
            ).get_image_bytes())

            generated_images.append(generated_image)
        except ValueError:
            generated_images.append(None)
    else:
        generated_images.append(None)            

df["generated_image"] = generated_images

if show_only_with_generated_image:
    view = df[df.generated_image.notnull()]
else:
    view = df


for _,row in view.iterrows():
    with st.container():
        st.write(f"## {row.caption}")
        col1, col2 = st.columns(2)
        col1.image(row.image)
        if row.generated_image is not None:
            col2.image(row.generated_image)





