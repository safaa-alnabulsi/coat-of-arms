import requests
import json
from PIL import Image
from io import BytesIO

# Map our dataset to the API keywords

LION_MODIFIERS_MAP = {
    'lion': 'lionRampant',
    'lion rampant': 'lionRampant',
    'lion passt': 'lionPassant',
    'lion passt guard': 'lionPassantGuardant',
    "lion's head": 'lionHeadCaboshed'
}

CROSS_MODIFIERS_MAP = {
    'cross': 'crossHummetty' ,
    'cross moline': 'crossMoline',
    'cross patonce': 'crossPatonce',
}

    
EAGLE_MODIFIERS_MAP = {
    'eagle': 'eagle' ,
    'eagle doubleheaded': 'eagleTwoHeards',
}


MODIFIERS_MAP = {**LION_MODIFIERS_MAP, **CROSS_MODIFIERS_MAP, **EAGLE_MODIFIERS_MAP}


COLORS_MAP = { 'A': 'argent', # silver
              'B': 'azure', # blue
              'O': 'or', # gold 
              'S': 'sable', # black
              'G': 'gules', # red
              'V': 'vert' # green
             }
#               'E':'',
#               'X': '', 
#               'Z': ''}


# Armoria-API = possible values for single position of a charge in an image
POSITION = ['a','b','c','d','e','f','g','h','i','y','z']

# Armoria-API = possible values for a charge scale 
# note that for large charges in a side positions means that a part of the charge is out of the shield
SCALE = ['0.5','1','1.5']

class ArmoriaAPIPayload:

    def __init__(self, struc_label, position='e', scale='1.5'):
        self.position = position
        self.scale = scale
        
        shield_color = struc_label['shield']['color']
        charge_color = struc_label['objects'][0]['color'] # for now, only first charge is considered
        charge = struc_label['objects'][0]['charge']
        first_modifier = struc_label['objects'][0]['modifiers'][0]
        
        try:
            self.api_shield_color = COLORS_MAP[shield_color]
            self.api_charge_color = COLORS_MAP[charge_color]
        except KeyError:
            raise ValueError('Invalid color')
        
        try:
            key = charge + ' ' + first_modifier
            self.api_charge = MODIFIERS_MAP[key]
        except KeyError:
            raise ValueError('Invalid charge')

    def get_armoria_payload(self):

        coa = {"t1": self.api_shield_color, 
           "shield":"heater",
           "charges":[{"charge": self.api_charge,
                       "t": self.api_charge_color,
                       "p": self.position, 
                       "size": self.scale}] 
          }

        return coa


class ArmoriaAPIWrapper:
    def __init__(self, size, format, coa):
        self.format = format
        payload = {"size": size, 
                   "format": format,
                   "coa": json.dumps(coa)
                  }
        self.r = requests.get('https://armoria.herokuapp.com/', params=payload)
#         print(self.r.url)
    
    def get_image_bytes(self):
        i = Image.open(BytesIO(self.r.content))
        return i
    
    def show_image(self):
        i = self.get_image_bytes()
        i.show()
    
    def save_image(self, full_path):
        i = self.get_image_bytes()    
        i.save(full_path, 'png')
  
