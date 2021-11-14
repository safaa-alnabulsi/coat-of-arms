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
                 weights_map=WEIGHT_MAP):
        self.predicted = Caption(predicted).get_structured()
        self.correct = Caption(correct).get_structured()
        self.weights_map= weights_map
        self.total_colors = 6 # from armoria_api.py
        self.total_charges = 3 # from armoria_api.py
        self.total_mods = 9 # counted from armoria_api.py ### maybe fix each ch with mod ==> count them together in next iteration
    
    def get(self):
        charge_score = self.get_charges_acc()
        shield_score = self.get_shield_acc()

        return (charge_score + shield_score) / 2

    def get_charges_acc(self):
        
        total_acc=[]
        
        for obj1 in self.correct['objects']:
            
            ch1    = obj1['charge']
            color1 = obj1['color']
            mods1  = obj1['modifiers']
            
            total= 2 + len(mods1)
            
            obj_acc=[]

            for obj2 in self.predicted['objects']:
                hits_ch_colors=0
                hits_charge=0
                hits_mod=0

                ch2    = obj2['charge']
                color2 = obj2['color']
                mods2  = obj2['modifiers']
                
                if ch1 == ch2:
                    hits_charge+= 1
                else:
                    continue
                
                if color1 == color2:
                    hits_ch_colors+= 1

                for cm in mods1:
                    for pm in mods2:
                        if cm == pm:
                            hits_mod+= 1
                            break

                hits = hits_ch_colors * self.weights_map['charge_color'] + \
                      hits_charge * self.weights_map['charge'] + \
                      hits_mod * self.weights_map['modifier']
                
                obj_acc.append(round(hits / total, 2))
                print(obj_acc)
                print('max(obj_acc) == ', max(obj_acc))
            
            if len(obj_acc) > 0:
                total_acc.append(max(obj_acc)) 
                
        # min, avg, max accuracy for each object in correct against all predicted
        
        print('predicted_cap: ', self.predicted)
        print('correct_cap: ', self.correct)
        
        if len(total_acc) == 0:
            return 0.0
            
        avg_acc = sum(total_acc)/len(total_acc)

        return avg_acc
                

    def get_shield_acc(self):
        hits_sh_colors=0
        hits_mod=0
        hits = 0
        total = 1
        
        color1 = self.predicted['shield']['color']
        mods1  = self.predicted['shield']['modifiers']
        color2 = self.correct['shield']['color']
        mods2  = self.correct['shield']['modifiers']

        if color1 == color2:
            hits_sh_colors+= 1

        for cm in mods1:
            total+=1
            for pm in mods2:
                if cm == pm:
                    hits_mod+= 1
                    break
        
        hits = hits_sh_colors * self.weights_map['shield_color'] + \
              hits_mod * self.weights_map['shield_mod']

        return round(hits / total, 2)
        