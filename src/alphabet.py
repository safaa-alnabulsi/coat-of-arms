COLORS = ['A', 'B', 'O', 'S', 'G', 'E', 'X', 'V', 'Z']
OBJECTS = ['lion', 'fess', 'cross', 'crescent', 'chief', 'chevron', 'escutch', 'escutcheon', 'mullet',
               'eagle', 'bars', 'fleur-de-lis', 'martlet', 'saltire', 'estoile', 'rose',
               'leave', 'fish', 'leopard', 'bear', 'ram', 'lamb', 'boar', 'cow', 'duck',
               'dragon', 'merlette', 'cock', 'volcano', 'falcon', 'wing', 'oak',
               'fleurs-de-lis', 'branch', 'bull', 'elephant', 'griffin', 'horseshoe', 'hare',
               'panther', 'hand', 'bugle-horn', 'lure', 'tail',
               'dice', 'donkey', 'face', 'unicorn', 'blackbird', 'marmite', 'attire']

MODIFIERS = ['rampant', 'dancetty', 'salient', 'roundel', 'bend', 'annulet', 'lozenge',
                 'orle', 'crusily', 'pale', 'doubleheaded', 'rising', 'addorsed', 'slipped',
                 'erect', 'cr.', 'chained', 'erased', 'hooded', 'winged', 'embattled',
                 'gorged', 'arched', 'segreant', 'pd', 'isst', 'jesst', 'passt', 'guard','passt guard', 'sejt', 'reguard',
                 'inv', 'cch', 'segr', 'p.c.', 'p.n.', 'col', 'displayed','sn', 'dx', 'dancetty', 'mount',
                 'flory', 'undy', 'masoned', 'bendy', 'potenty', 'checky', 
                 'compony', 'roundely', 'engrailed', 'crenelated', 'nebuly', 'castely', 'moline','patonce', 'head', 'heads']

# ---------------------------------------------------------------------------
# extracted from dataset & compared with https://github.com/Azgaar/armoria-api/blob/main/app/dataModel.js#L8

SHIELD_MODIFIERS = ['border', 'checky', 'potenty', 'bendy', 'masoned', 'vairy', 'flory', 'undy',
                    'compony', 'roundely', 'engrailed', 'crenelated', 'nebuly', 'castely']

# our dataset <> armoria

# engrailed <> engrailed
# nebuly <> nebuly
# masoned <> masoned
# potenty <>  potenty, potentyDexter, potentySinister
# bendy   <>  bendy, bendySinister, palyBendy, barryBendy

# checky  <>  chequy
# roundely <> roundel, roundel2

# vairy <> -
# flory <> -
# undy <> -
# crenelated <> -
# castely <> -
# ---------------------------------------------------------------------------


POSITIONS = ['acc.', 'ch.', 'in', 'of', '&', 'and', 'above', 'with',
                 'betw', 'indented', 'towards', 'chf', 'on'
                 ]
NUMBERS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10','11']

SIMPLE_OBJECTS = ['lion', 'cross', 'head', 'eagle']
SIMPLE_COLORS = ['G', 'S'] # G: red , S: black
SIMPLE_MODIFIERS = ['rampant', 'dancetty', 'moline', 'patonce', 'passt guard', 'passt', 'guard']