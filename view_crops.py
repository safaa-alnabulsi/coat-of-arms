import streamlit as st
from copy import deepcopy as dc
import pandas as pd
from pathlib import Path

from PIL import Image
import numpy as np

from src.label_checker_automata import LabelCheckerAutomata
from src.armoria_api import ArmoriaAPIPayload, ArmoriaAPIWrapper
from src.caption import Caption

DEBUG = False



@st.cache
def load_data():
    
    data_dir = Path("data/cropped_coas/out")
    # else:
    #     data_dir = Path("data/cropped_coas/out_valid")

    images, captions = [], []

    for image_fn in data_dir.iterdir():
        if image_fn.suffix == ".jpg" and not image_fn.name.startswith("."):
            image = Image.open(image_fn)
            image.thumbnail((150,150))
            image = np.asarray(image)
            images.append(image)
            
            captions.append("_".join(image_fn.stem.split("_")[1:]))

    df = pd.DataFrame.from_dict({
        "image": images,
        "caption": captions,
    })

    if DEBUG:
        df = pd.DataFrame(df.sample(n=10))

    generated_images = []

    for irow, row in df.iterrows():

        caption = Caption(row.caption, support_plural=False)
        if caption.is_valid:
            try:
                armoria_payload = caption.get_armoria_payload_dict()
                generated_image = np.asarray(ArmoriaAPIWrapper(
                    size=150,
                    format="png", 
                    coa=armoria_payload
                ).get_image_bytes())

                generated_images.append(generated_image)
            except ValueError:
                generated_images.append(None)
        else:
            generated_images.append(None)            

    df["generated_image"] = generated_images

    return df


df = load_data()

df = dc(df)


with st.container():
    st.write(f"""## Data description
    * {len(df)} images
    * {df.generated_image.notnull().sum()} images with armoria generated image"""
    )

df = pd.DataFrame(df[df.generated_image.notnull()])

@st.cache
def get_caption_data(df):
    result = []
    automata = LabelCheckerAutomata()
    for _, row in df.iterrows():
        parsed_label = automata.parse_label(row.caption)
        aligned = automata.align_parsed_label(row.caption, parsed_label)
        result.append(aligned)
    return pd.DataFrame(result)


df.reset_index(inplace=True)
caption_data = get_caption_data(df)
df = pd.concat([df, caption_data], axis=1)

all_colors = sorted(list(set([it for ll in caption_data.colors.to_list() for it in ll])))
all_objects = sorted(list(set([it for ll in caption_data.objects.to_list() for it in ll])))
all_modifiers = sorted(list(set([it for ll in caption_data.modifiers.to_list() for it in ll])))

selected_colors = st.sidebar.multiselect(
    'colors',
    all_colors,
    all_colors
)
selected_objects = st.sidebar.multiselect(
    'objects',
    all_objects,
    all_objects
)
selected_modifiers = st.sidebar.multiselect(
    'modifiers',
    all_modifiers,
    all_modifiers
)

view = df[np.logical_and.reduce([
    df.colors.apply(lambda color_list: len(set(color_list) - set(selected_colors)) == 0 ),
    df.objects.apply(lambda object_list: len(set(object_list) & set(selected_objects))>0),
    df.modifiers.apply(lambda modifier_list: len(set(modifier_list) & set(selected_modifiers))>0),
])]


if len(view)>10:
    len_current_filter_set = len(view)
    view = view.sample(n=10)
    st.write("## !reduce result set to maximum of 10 samples!")
    st.write(f"The current filter set actually returned {len_current_filter_set} samples")

for _,row in view.iterrows():
    with st.container():
        st.write(f"## {row.caption}")
        col1, col2 = st.columns(2)
        col1.image(row.image)
        if row.generated_image is not None:
            col2.image(row.generated_image)
