from src.caption import Caption

WEIGHT_MAP = {'shield_color': 1, 'shield_mod': 1, 'charge_color': 1, 'charge': 1, 'modifier': 1}

WEIGHT_MAP_OLD = {'shield_color': 10, 'charge_color': 10, 'charge': 60, 'modifier': 20}

# Notes:
# acc = number of True answers / number of samples

# a = np.ones(10)
# b = np.ones(10)
# (a == b).sum() / len(a)

class Accuracy:
    
    def __init__(self, predicted, correct, 
                 weights_map=WEIGHT_MAP_OLD):
        self.predicted = predicted
        self.correct = correct
        self.weights_map= weights_map
        self.total_colors = 6 # from armoria_api.py
        self.total_charges = 3 # from armoria_api.py
        self.total_mods = 9 # counted from armoria_api.py ### maybe fix each ch with mod ==> count them together in next iteration
    
    def get(self):
        predicted_cap = Caption(self.predicted).get_aligned()
        correct_cap = Caption(self.correct).get_aligned()
        
        acc_sh_colors=0
        acc_ch_colors=0
        acc_charge=0
        acc_mod=0
        acc = 0

        ## TODO fix the align_parsed_label() function to return shield color and charges colors
        try:
            if predicted_cap['colors'][0] == correct_cap['colors'][0]:
                acc_sh_colors+= 1
            if predicted_cap['colors'][1] == correct_cap['colors'][1]:
                acc_ch_colors+= 1
        except:
            pass
#             print('error in accessing key of predicted_cap',len(predicted_cap['colors']))

        for co in correct_cap['objects']:
            for po in predicted_cap['objects']:
                if co == po:
                    acc_charge+= 1
                    break
                    

        for cm in correct_cap['modifiers']:
            for pm in predicted_cap['modifiers']:
                if cm == pm:
                    acc_mod+= 1
                    break
        
        acc = acc_sh_colors * self.weights_map['shield_color'] + \
              acc_ch_colors * self.weights_map['charge_color'] + \
              acc_charge * self.weights_map['charge'] + \
              acc_mod * self.weights_map['modifier']

        return acc

    def get_charges_acc(self):
        
        acc_ch_colors=0
        acc_charge=0
        acc_mod=0
        acc = 0
        
        predicted_cap = Caption(self.predicted).get_structured()
        correct_cap = Caption(self.correct).get_structured()
        
        for ch1, color1, mods1 in predicted_cap['objects']:
            for ch2, color2, mods2 in predicted_cap['objects']:
                if ch1 == ch2:
                    acc_charge+= 1
                
                if color1 == color2:
                    acc_ch_colors+= 1
                
                for cm in mods1:
                    for pm in mods1:
                        if cm == pm:
                            acc_mod+= 1
                            break

        acc = acc_ch_colors * self.weights_map['charge_color'] + \
              acc_charge * self.weights_map['charge'] + \
              acc_mod * self.weights_map['modifier']

        return acc
                

    def get_shield_acc(self):
        acc_sh_colors=0
        acc_mod=0
        acc = 0

        predicted_cap = Caption(self.predicted).get_structured()
        correct_cap = Caption(self.correct).get_structured()
        
        color1 = predicted_cap['shield']['color']
        mods1  = predicted_cap['shield']['modifiers']
        color2 = correct_cap['shield']['color']
        mods2  = correct_cap['shield']['modifiers']

        if color1 == color2:
            acc_sh_colors+= 1

        for cm in mods1:
            for pm in mods1:
                if cm == pm:
                    acc_mod+= 1
                    break
        
        acc = acc_sh_colors * self.weights_map['shield_color'] + \
              acc_mod * self.weights_map['shield_mod']

        return acc
        