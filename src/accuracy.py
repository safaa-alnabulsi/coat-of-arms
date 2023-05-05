from itertools import permutations
from src.caption import Caption

WEIGHT_MAP = {'shield_color': 1, 'shield_mod': 0, 'charge_color': 1, 'charge': 1, 'modifier': 1}

# WEIGHT_MAP_OLD = {'shield_color': 10, 'charge_color': 10, 'charge': 60, 'modifier': 20}

WEIGHT_MAP_ONLY_CHARGE = {'shield_color': 0, 'shield_mod': 0, 'charge_color': 0, 'charge': 1, 'modifier': 1}

WEIGHT_MAP_ONLY_CHARGE_COLOR = {'shield_color': 0, 'shield_mod': 0, 'charge_color': 1, 'charge': 0, 'modifier': 0}

WEIGHT_MAP_ONLY_SHIELD_COLOR = {'shield_color': 1, 'shield_mod': 0, 'charge_color': 0, 'charge': 0, 'modifier': 0}

# Notes:
# acc = number of True answers / number of samples

# a = np.ones(10)
# b = np.ones(10)
# (a == b).sum() / len(a)

class AccuracyColor:
    
     def __init__(self, predicted, correct):
            self.predicted = predicted.split(" ")
            self.correct = correct.split(" ")
            self.total_colors = 6 # from armoria_api.py
     
     def get(self):
        hits = 0
        correct_shield_color = self.correct[0]
        correct_ch_color = self.correct[1]

        predicted_shield_color = self.predicted[0]
        
        try:
            predicted_ch_color = self.predicted[1]
        except:
            predicted_ch_color = ''
            print('no color prediction')  
            
        if correct_shield_color == predicted_shield_color:
            hits+=1
            
        if correct_ch_color == predicted_ch_color:
            hits+=1
        
        return hits / 2
    
class Accuracy:
    
    def __init__(self, predicted, correct, 
                 weights_map=WEIGHT_MAP):
        self.predicted = Caption(predicted, support_plural=True).get_structured()
        self.correct = Caption(correct, support_plural=True).get_structured()
        self.weights_map= weights_map
        self.total_colors = 6 # from armoria_api.py
        self.total_charges = 3 # from armoria_api.py
        self.total_mods = 9 # counted from armoria_api.py ### maybe fix each ch with mod ==> count them together in next iteration
    
    def get(self):

        shield_score = self.get_shield_acc()
        if self.weights_map == WEIGHT_MAP_ONLY_SHIELD_COLOR:
            return shield_score
            
#         charge_score, charge_color_score = self.get_charges_acc()
        charge_score, charge_color_score = self.get_single_charges_acc()

        if self.weights_map == WEIGHT_MAP_ONLY_CHARGE:
            return charge_score

        if self.weights_map == WEIGHT_MAP_ONLY_CHARGE_COLOR:
            return charge_color_score

        # we devivde here by 3 and not 4, as we condier the charge and its modiferes one single entity 
        return round((charge_score + charge_color_score + shield_score) / 3 , 2) 

    # we conisder having one object in each caption
    def get_single_charges_acc(self):  
        
        
        hits_ch_colors=0
        hits_charge=0
        hits_mod=0

        
        obj1 = self.correct['objects'][0]
        ch1    = obj1['charge']
        color1 = obj1['color']
        mods1  = obj1['modifiers']
        
        # default value is when having WEIGHT_MAP
        # we drop the color of the charge here, we don't care about it
        # this means that we start from correct caption and we only care about correct modifiers 
        # which mean we don't penalize predicting extra modifier  
        total = len(mods1) + 1

        if len(self.predicted['objects']) == 0:
            return 0,0
        
        obj2 = self.predicted['objects'][0]

        ch2    = obj2['charge']
        color2 = obj2['color']
        mods2  = obj2['modifiers']


        if color1.lower() == color2.lower():
            hits_ch_colors+= 1

        if ch1 == ch2:
            hits_charge+= 1

        for cm in mods1:
            for pm in mods2:
                if cm == pm:
                    hits_mod+= 1
                    break

        charge_hits = hits_charge * self.weights_map['charge'] + hits_mod * self.weights_map['modifier']

        charge_color_hits = hits_ch_colors * self.weights_map['charge_color']

        avg_charge_acc = round(charge_hits / total, 2)
        avg_charge_color_acc = round(charge_color_hits, 2)
       

        return avg_charge_acc, avg_charge_color_acc

    def get_charges_acc(self):
        # The task is to find compinations which bring us the maximum total accuracy 
        
        total_acc=[]
       
        all_obj_acc = []
        all_obj_color_acc = []
     
        for obj1 in self.correct['objects']:
            
            ch1    = obj1['charge']
            color1 = obj1['color']
            mods1  = obj1['modifiers']
            
            # default value is when having WEIGHT_MAP
            # we drop the color of the charge here, we don't care about it
            # this means that we start from correct caption and we only care about correct modifiers 
            # which mean we don't penalize predicting extra modifier  
            total = len(mods1) + 1
           
            obj_acc=[]
            obj_color_acc=[]

            for obj2 in self.predicted['objects']:
                hits_ch_colors=0
                hits_charge=0
                hits_mod=0

                ch2    = obj2['charge']
                color2 = obj2['color']
                mods2  = obj2['modifiers']
                
                if color1.lower() == color2.lower():
                    hits_ch_colors+= 1

                if ch1 == ch2:
                    hits_charge+= 1
                # else: # cannot remember why I added this condition here
                #     continue
                
                for cm in mods1:
                    for pm in mods2:
                        if cm == pm:
                            hits_mod+= 1
                            break

                charge_hits = hits_charge * self.weights_map['charge'] + \
                      hits_mod * self.weights_map['modifier']
                
                charge_color_hits = hits_ch_colors * self.weights_map['charge_color']
                   
                
                obj_acc.append(round(charge_hits / total, 2))
                obj_color_acc.append(round(charge_color_hits, 2))

            if len(obj_acc) > 0:
                all_obj_acc.append(obj_acc)
                
            if len(obj_color_acc) > 0:
                all_obj_color_acc.append(obj_color_acc)
                
        # min, avg, max accuracy for each object in correct against all predicted
        
#         print('predicted_cap: ', self.predicted)
#         print('correct_cap: ', self.correct)
        
        
        if len(all_obj_acc) != 0 and len(all_obj_color_acc)!=0:
            _, avg_charge_acc = self.get_max_accuracy(all_obj_acc)
            _, avg_charge_color_acc = self.get_max_accuracy(all_obj_color_acc)

            return avg_charge_acc, avg_charge_color_acc

        return 0.0, 0.0
                

    # the new real dataset doesn't have shield modifier "border" 
    # Hence, I'm commenting out the modifier calculation
    def get_shield_acc(self):
        hits_sh_colors=0
        hits_mod=0
        hits = 0
        total = 1
        
        try:
            color1 = self.correct['shield']['color']
#             mods1  = self.correct['shield']['modifiers']
        except KeyError as err:
            print(f"KeyError - Unexpected {err=}, {type(err)=} in correct label: {self.predicted=}")
            raise err
            
        try:
            color2 = self.predicted['shield']['color']
#             mods2  = self.predicted['shield']['modifiers']
        except KeyError as err:
            print(f"KeyError - Unexpected {err=}, {type(err)=} in predicted label: {self.correct=}")
            raise err
            

        if color1.lower() == color2.lower():
            hits_sh_colors+= 1

#         for cm in mods1:
#             total+=1
#             for pm in mods2:
#                 if cm == pm:
#                     hits_mod+= 1
#                     break
        
        hits = hits_sh_colors * self.weights_map['shield_color'] #+ hits_mod * self.weights_map['shield_mod']
              
        
        # we drop the number of modifier here and the charge itself
        if self.weights_map == WEIGHT_MAP_ONLY_SHIELD_COLOR:
            total = 1

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

        print('comblist', comblist)
        print('all_values', all_values)
        print('max_index', max_index)
        print('max_acc', max_acc)

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
            
