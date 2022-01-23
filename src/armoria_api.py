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

SHIELD_MODIFIERS_MAP = {
    'border': 'bordure' ,
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

# chequy
# Armoria-API = possible values for single position of a charge in an image
# SINGLE_POSITION = ['a','b','c','d','e','f','g','h','i','y','z']

POSITIONS ={1: 'e',
            2: 'kn',   
            3: 'def',
            4: 'bhdf',
            5: 'kenpq',
            6: 'kenpqa',
            7: 'kenpqac',   
            8: 'abcdfgzi',
            9: 'abcdefgzi',
            11: 'ABCDEFGHIJKL'    
            }

# Armoria-API = possible values for a charge scale 
# note that for large charges in a side positions means that a part of the charge is out of the shield
SCALE = ['0.5','1','1.5']

# =====================================================================================

class ArmoriaAPIPayload:

    def __init__(self, struc_label, position='e', scale='1.5'):
        self.position = position
        self.scale = scale
        
        # Shield
        shield_color = struc_label['shield']['color'].upper() # dict keys are in upper case
        
        try:
            self.api_shield_color = COLORS_MAP[shield_color] 
        except KeyError:
            raise ValueError('Invalid shield_color', shield_color)

            
        self.charges = self._get_charges(struc_label['objects'])
            
        self.ordinaries = self._get_ordinaries(struc_label['shield']['modifiers'])

    
    def get_armoria_payload(self):
        coa = {"t1": self.api_shield_color, 
           "shield":"heater",
           "charges": self.charges,
           "ordinaries": self.ordinaries
          }

        return coa
        
    # private_func(( border for now ))
    def _get_ordinaries(self, shield_modifiers):
        ordinaries = []
        
        for mod in shield_modifiers:
            
            try:
                api_shield_modifier = SHIELD_MODIFIERS_MAP[mod]
                ordinary = {"ordinary":api_shield_modifier, "t":"azure"}
                ordinaries.append(ordinary)
                
            except KeyError:
                raise ValueError('Invalid ordinary')
                
        return ordinaries
        
    # private_func
    def _get_charges(self, objects):
        
        charges = []
              
        for obj in objects:

            try:
                charge_color = obj['color'].upper()
                api_charge_color = COLORS_MAP[charge_color]
            except KeyError:
                raise ValueError('Invalid charge_color', charge_color)
           
            try:
                charge = obj['charge']
            except IndexError:
                raise ValueError('Invalid charge')
            
            try:
                first_modifier = ' ' + obj['modifiers'][0]
            except IndexError:
                first_modifier = ''

            try:
                key = charge + first_modifier
                api_charge = MODIFIERS_MAP[key]
            except KeyError:
                raise ValueError('Invalid charge')
                
            charge = {"charge": api_charge,
                       "t": api_charge_color,
                       "p": self.position,  
                       "size": self.scale}
            
            charges.append(charge)
            
        return charges
                 
# =====================================================================================

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
  
