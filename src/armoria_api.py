import requests
import json
from PIL import Image
from io import BytesIO

# Map our dataset to the API keywords

LION_MODIFIERS_MAP = {
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
    
    def __init__(self, label):
        shield_color = label[0]
        charge_color = label[1]
        charge = label[2]
        try:
            self.position = label[3]
            self.scale = label[4]
        except:
            self.position = 'e'
            self.scale = '1.5'

        self.api_shield_color = COLORS_MAP[shield_color]
        self.api_charge_color = COLORS_MAP[charge_color]
        self.api_charge = MODIFIERS_MAP[charge]

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
  
