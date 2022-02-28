import requests
import json
from PIL import Image
from io import BytesIO

# Map our dataset to the API keywords
# Lion
SINGLE_LION_MODIFIERS_MAP = {
    'lion': 'lionRampant',
    'lion rampant': 'lionRampant',
    'lion passt': 'lionPassant',
    'lion passt guard': 'lionPassantGuardant',
    "lion's head": 'lionHeadCaboshed',
}
PLURAL_LION_MODIFIERS_MAP = {
    'lions': 'lionRampant',
    'lions rampant': 'lionRampant',
    'lions passt': 'lionPassant',
    'lions passt guard': 'lionPassantGuardant',
}
LION_MODIFIERS_MAP ={**SINGLE_LION_MODIFIERS_MAP, **PLURAL_LION_MODIFIERS_MAP}

# Eagle
SINGLE_EAGLE_MODIFIERS_MAP = {
    'eagle': 'eagle' ,
    'eagle doubleheaded': 'eagleTwoHeards',
}
PLURAL_EAGLE_MODIFIERS_MAP = {
    'eagles': 'eagle' ,
    'eagles doubleheaded': 'eagleTwoHeards' ,
}
EAGLE_MODIFIERS_MAP = {**SINGLE_EAGLE_MODIFIERS_MAP, **PLURAL_EAGLE_MODIFIERS_MAP}

CROSS_MODIFIERS_MAP = {
    'cross': 'crossHummetty' ,
    'cross moline': 'crossMoline',
    'cross patonce': 'crossPatonce',
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
SINGLE_POSITION = ['a','b','c','d','e','f','g','h','i','y','z']

POSITIONS ={'1': 'e',
            '2': 'kn',   
            '3': 'def',
            '4': 'bhdf',
            '5': 'kenpq',
            '6': 'kenpqa',
            '7': 'kenpqac',   
            '8': 'abcdfgzi',
            '9': 'abcdefgzi',
            '11': 'ABCDEFGHIJKL'    
            }

# Armoria-API = possible values for a charge scale 
# note that for large charges in a side positions means that a part of the charge is out of the shield
SIZES ={'1': '1.5',
        '2': '0.7',   
        '3': '0.5',
        '4': '0.5',
        '5': '0.5',
        '6': '0.5',
        '7': '0.5',   
        '8': '0.3',
        '9': '0.3',
        '11': '0.18'    
        }

NUMBERS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
# maximum allowed number of objects in one coat-of-arm is 11
NUMBERS_MULTI = ['2', '3', '4', '5']

# =====================================================================================

class ArmoriaAPIPayload:

    def __init__(self, struc_label, position='e', scale='1.5'):
        print(struc_label)
        self.position = position
        self.objects = struc_label['objects']
        self.charge_positions = self._get_positions()
        self.scale = self._get_charges_size()

        # Shield
        shield_color = struc_label['shield']['color'].upper() # dict keys are in upper case
        
        try:
            self.api_shield_color = COLORS_MAP[shield_color] 
        except KeyError:
            raise ValueError('Invalid shield_color', shield_color)

            
        self.charges = self._get_charges(self.objects)
            
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

        for i, obj in enumerate(objects):
            try:
                charge_color = obj['color'].upper()
                api_charge_color = COLORS_MAP[charge_color]
            except KeyError:
                raise ValueError('Invalid charge_color', charge_color)
                       
            try:
                first_modifier = ' ' + obj['modifiers'][0]
            except IndexError:
                first_modifier = ''

            try:
                key = obj['charge'] + first_modifier
                api_charge = MODIFIERS_MAP[key]
            except KeyError:
                raise ValueError('Invalid charge', key)
             
            charge = {"charge": api_charge,
                       "t": api_charge_color ,
                       "p": self.charge_positions[i],  
                       "size": self.scale}
            
            charges.append(charge)
            
        return charges
  
    # This function supports single, plural and multi objects
    # the ideais to count total number of final charges and connect them back to charges
    # example: 3 lions 3 eagles: two charges and 6 final drawn objs on the coa => kenpqa for all objects
    # which means ken for 3 lions and pqa for 3 eagles
    def _get_charges_positions(self):
        total_obj_number = self._get_total_objects_number()
        
        try:
            pos = POSITIONS[str(total_obj_number)]   
        except KeyError:
            raise ValueError('Invalid number of charge', total_obj_number)

        return pos        
    
    def _get_positions(self):
        positions = []
        pos = self._get_charges_positions()
        start_index_pos = 0

        # single object
        if len(self.objects) == 1:
            return [pos]

        # multi object
        for obj in self.objects:
            end_index_pos = int(obj['number']) + start_index_pos 
            charge_position = pos[start_index_pos: end_index_pos]
            positions.append(charge_position)
            start_index_pos = end_index_pos

        return positions

    def _get_charges_size(self):
        total_obj_number = self._get_total_objects_number()
        try:
            size = SIZES[str(total_obj_number)]   
        except KeyError:
            raise ValueError('Invalid number of charge', total_obj_number)

        return size        

    def _get_total_objects_number(self):
        total_obj_number = 0  
        for obj in self.objects:
            obj_num = obj['number']
            total_obj_number+=int(obj_num)

        return total_obj_number

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
  
