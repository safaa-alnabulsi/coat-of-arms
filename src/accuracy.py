from itertools import permutations
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
        # The task is to find compinations which bring us the maximum total accuracy 
        
        total_acc=[]
        
        all_obj_acc = []
        
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
            
            if len(obj_acc) > 0:
                all_obj_acc.append(obj_acc)
                
        # min, avg, max accuracy for each object in correct against all predicted
        
#         print('predicted_cap: ', self.predicted)
#         print('correct_cap: ', self.correct)
        
        if len(all_obj_acc) == 0:
            return 0.0
        
        _, avg_acc = self.get_max_accuracy(all_obj_acc)

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

    
    def get_max_accuracy(self, all_obj_acc):
#         print('all_obj_acc', all_obj_acc)
        n = len(all_obj_acc[0])
        # get all possible unique combinations of object indexes
        comblist = self.generate_all_permutations(n)
        # calculate the accuracy sum of each combination 
        all_values = self.get_all_values(comblist, all_obj_acc)
        # get the maximum sum and the indexs 
        max_index, max_acc = self.get_max_accuracy_item(all_values)

#         print('comblist', comblist)
#         print('all_values', all_values)
#         print('max_index', max_index)
#         print('max_acc', max_acc)

        return max_index, max_acc
            
            
    # -------- functions below are used in calculating the maximum accuracy: get_max_accuracy ------- #
    
    def generate_all_permutations(self, n):
        l = [i for i in range(0, n)]
        comb = permutations(l, n)
        comblist = [list(i) for i in comb]
        
        return comblist
            
    def get_all_values(self, comblist, all_obj_acc):
        all_values = []
        for item in comblist:       
            acc = self.get_comb_acc(item, all_obj_acc)
            all_values.append({'index': item, 'value:': acc})
            
        return all_values

    def get_comb_acc(self, item, all_obj_acc):
        total_sum = 0
        n = len(all_obj_acc)
        for i in range(0, n):
            l = all_obj_acc[i]
            try:
                total_sum += l[item[i]] 
            except IndexError as e:
                print('Index {} does not exist in item, seems like predicted less than Ground truth'.format(i)) 

        return round(total_sum/ n, 2)

    def get_max_accuracy_item(self, all_values):
        max_acc = -1
        max_index = []
        for comb_val in all_values:  
            t1, t2 = comb_val.items()
            index, value = t1[1], float(t2[1])
            if value > max_acc:
                max_acc = value
                max_index = index
                
        return max_index, max_acc
        
    # -------- functions below are used in calculating the maximum accuracy ------- #
            
