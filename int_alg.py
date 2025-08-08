import numpy as np
from mine import mine, mine_fa
import math
import os
import pickle

class int_alg_per_varb:
    def __init__(self, obs, acs, chosen_varb, mines, mines_std, num_iterations, scalable=True):
        self.num_iterations = num_iterations
        print(scalable, 'scalable in func')
        self.mines = mines
        self.mines_std = mines_std
        self.chosen_varb = chosen_varb
        self.obs = obs.round(decimals=2)
        self.acs = acs.round(decimals=2)
        self.expts = 3
        self.synergistic = []
        self.redundants = []
        self.reds_n_norms = []
        self.redundant_reductions = []
        self.redundant_reductions_std = []
        self.means = []
        # When scalable is True, use mine_fa; otherwise use mine.
        self.mi_method = "mine_fa" if scalable else "mine"

    def do_mine(self, data, acs, num_iterations):
        if self.mi_method == "mine":
            return self._mine_estimate(data, acs, num_iterations)
        else:
            return self._mine_fa_estimate(data, acs, num_iterations)

    def _mine_estimate(self, data, acs, num_iterations):
        # Returns the computed value using the mine function.
        return mine(data, acs, num_iterations).run()

    def _mine_fa_estimate(self, data, acs, num_iterations):
        # Returns the computed value using the mine_fa function.
        return mine_fa(data, acs, num_iterations).run()

    def get_theta(self, mathcal_x):
        results_not_inc = self.do_mine(self.obs[:, mathcal_x], self.acs, self.num_iterations)
        mathcal_x_big = self.append_element(mathcal_x, self.chosen_varb)
        results_inc = self.do_mine(self.obs[:, mathcal_x_big], self.acs, self.num_iterations)
        print('this is mi including chosen varb', results_inc.mean(axis=0)[-1],
              'this is mi not including chosen varb', results_not_inc.mean(axis=0)[-1])
        theta_means = (results_inc - results_not_inc).mean(axis=0)[-1]
        theta_std = (results_inc - results_not_inc).std(axis=0)[-1]
        return theta_means, theta_std

    def append_element(self, original_list, element):
        new_list = original_list.copy()
        new_list.append(element)
        return new_list

    def run_mi_varb(self):
        indexes = list(range(self.obs.shape[1] - 1, -1, -1))
        obs_run_varb = self.obs
        del indexes[-(self.chosen_varb + 1)]
        for ob_num in indexes:
            results = self.do_mine(self.obs[:, ob_num], self.obs[:, self.chosen_varb], self.num_iterations)
            self.means.append(results.mean(axis=0)[-1])
        self.keepers = [x for _, x in sorted(zip(self.means, indexes))][::-1]
        print('Here are the mis', np.array(self.means), 'and heres the indexes', self.keepers)

    def run_syn_varb(self):
        self.mi_with_ob_num, self.mi_with_ob_num_std = self.get_theta(self.keepers)
        for ob_num in self.keepers:
            new_keepers = [item for item in self.keepers if item != ob_num]
            mi_without_ob_num, mi_without_ob_num_std = self.get_theta(new_keepers)
            print('this is the decrease in uncertainty caused by adding the target feature without its comparater',
                  self.mi_with_ob_num, 'std', self.mi_with_ob_num_std,
                  'heres without', mi_without_ob_num, 'std', mi_without_ob_num_std)
            if self.mi_with_ob_num + 2 * self.mi_with_ob_num_std / math.sqrt(self.expts) < \
               mi_without_ob_num - 2 * mi_without_ob_num_std / math.sqrt(self.expts):
                print(f'Variable {ob_num} contains net-redundant information with respect too {self.chosen_varb}')
                self.keepers = new_keepers
                self.redundants.append(ob_num)
                self.redundant_reductions.append(self.mi_with_ob_num - mi_without_ob_num)
                self.redundant_reductions_std.append(mi_without_ob_num_std)
                self.mi_with_ob_num = mi_without_ob_num
                self.mi_with_ob_num_std = mi_without_ob_num_std
            elif self.mi_with_ob_num - 2 * self.mi_with_ob_num_std / math.sqrt(self.expts) > \
                 mi_without_ob_num + 2 * mi_without_ob_num_std / math.sqrt(self.expts):
                print(f'Variable {ob_num} contains net-synergistic information with respect too {self.chosen_varb}')
                self.synergistic.append(ob_num)
            else:
                print(f'Variable {ob_num} contains no net-synergistic information with respect too {self.chosen_varb}')
                self.keepers = new_keepers
                self.mi_with_ob_num = mi_without_ob_num
                self.mi_with_ob_num_std = mi_without_ob_num_std
                self.reds_n_norms.append(ob_num)

        if len(self.synergistic) > 0:
            syn, self.syn_std = self.get_theta(self.synergistic)
            self.net_synergy = syn - self.mines[self.chosen_varb][-1]
        else:
            self.net_synergy = 0
            self.syn_std = 0

    def run_varb(self):
        self.run_mi_varb()
        self.run_syn_varb()


class int_alg_per_varb_mnist:
    def __init__(self, obs, acs, chosen_varb, mines, mines_std, num_iterations, scalable=True):
        self.num_iterations = num_iterations
        self.mines = mines
        self.mines_std = mines_std
        self.chosen_varb = chosen_varb
        self.obs = obs.round(decimals=2)
        self.acs = acs.round(decimals=2)
        self.expts = 3
        self.synergistic = []
        self.redundants = []
        self.reds_n_norms = []
        self.redundant_reductions = []
        self.redundant_reductions_std = []
        self.means = []
        self.mi_method = "mine_fa" if scalable else "mine"

    def do_mine(self, data, acs, num_iterations):
        if self.mi_method == "mine":
            return mine(data, acs, num_iterations).run()
        else:
            return mine_fa(data, acs, num_iterations).run()

    def get_theta(self, mathcal_x):
        results_not_inc = self.do_mine(self.obs[:, mathcal_x], self.acs, self.num_iterations)
        mathcal_x_big = self.append_element(mathcal_x, self.chosen_varb)
        results_inc = self.do_mine(self.obs[:, mathcal_x_big], self.acs, self.num_iterations)
        print('this is mi including chosen varb', results_inc.mean(axis=0)[-1],
              'this is mi not including chosen varb', results_not_inc.mean(axis=0)[-1])
        theta_means = (results_inc - results_not_inc).mean(axis=0)[-1]
        theta_std = (results_inc - results_not_inc).std(axis=0)[-1]
        return theta_means, theta_std

    def append_element(self, original_list, element):
        new_list = original_list.copy()
        new_list.append(element)
        return new_list

    def run_mi_varb(self):
        indexes = list(range(self.obs.shape[1] - 1, -1, -1))
        obs_run_varb = self.obs
        del indexes[-(self.chosen_varb + 1)]
        for ob_num in indexes:
            results = self.do_mine(self.obs[:, ob_num], self.obs[:, self.chosen_varb], self.num_iterations)
            self.means.append(results.mean(axis=0)[-1])
        self.keepers = [x for _, x in sorted(zip(self.means, indexes))][::-1]
        print('Here are the mis', np.array(self.means), 'and heres the indexes', self.keepers)

    def run_syn_varb(self):
        self.mi_with_ob_num, self.mi_with_ob_num_std = self.get_theta(self.keepers)
        for ob_num in self.keepers:
            new_keepers = [item for item in self.keepers if item != ob_num]
            mi_without_ob_num, mi_without_ob_num_std = self.get_theta(new_keepers)
            print('this is the decrease in uncertainty caused by adding the target feature without its comparater',
                  self.mi_with_ob_num, 'std', self.mi_with_ob_num_std,
                  'heres without', mi_without_ob_num, 'std', mi_without_ob_num_std)
            if self.mi_with_ob_num + 2 * self.mi_with_ob_num_std / math.sqrt(self.expts) < \
               mi_without_ob_num - 2 * mi_without_ob_num_std / math.sqrt(self.expts):
                print(f'Variable {ob_num} contains net-redundant information with respect to {self.chosen_varb}')
                self.keepers = new_keepers
                self.redundants.append(ob_num)
                self.redundant_reductions.append(self.mi_with_ob_num - mi_without_ob_num)
                self.redundant_reductions_std.append(mi_without_ob_num_std)
                self.mi_with_ob_num = mi_without_ob_num
                self.mi_with_ob_num_std = mi_without_ob_num_std
            elif self.mi_with_ob_num - 2 * self.mi_with_ob_num_std / math.sqrt(self.expts) > \
                 mi_without_ob_num + 2 * mi_without_ob_num_std / math.sqrt(self.expts):
                print(f'Variable {ob_num} contains net-synergistic information with respect to {self.chosen_varb}')
                self.synergistic.append(ob_num)
            else:
                print(f'Variable {ob_num} contains no net-synergistic information with respect to {self.chosen_varb}')
                self.keepers = new_keepers
                self.mi_with_ob_num = mi_without_ob_num
                self.mi_with_ob_num_std = mi_without_ob_num_std
                self.reds_n_norms.append(ob_num)

        if len(self.synergistic) > 0:
            syn, self.syn_std = self.get_theta(self.synergistic)
            self.net_synergy = syn - self.mines[0][-1]
        else:
            self.net_synergy = 0
            self.syn_std = 0

    def run_varb(self):
        self.run_mi_varb()
        self.run_syn_varb()



class int_alg:
    def __init__(self, obs, acs, num_iterations, scalable=True):
        self.num_iterations = num_iterations
        self.scalable = scalable
        self.obs = np.nan_to_num(obs).round(decimals=2)
        self.acs = np.nan_to_num(acs).round(decimals=2)
        self.mines = []
        self.mines_std = []
        self.store_data = []
        self.store_data_std = []
        self.store_syns_n_reds = []
        self.expts = 5
        self.mi_method = "mine_fa" if scalable else "mine"

    def do_mine(self, data, acs, num_iterations):
        if self.mi_method == "mine":
            return mine(data, acs, num_iterations).run()
        else:
            return mine_fa(data, acs, num_iterations).run()

    def get_mis(self):
        print('Getting MI')
        for i in range(self.obs.shape[1]):
            results = self.do_mine(self.obs[:, i], self.acs, self.num_iterations)
            means = results.mean(axis=0)
            std = results.std(axis=0)
            self.mines.append(means)
            self.mines_std.append(std)
            print(f'MI for varb {i} = {means[-1]} +/- {std[-1]*2/math.sqrt(self.expts)}')

    def run(self):
        self.get_mis()
        for chosen_varb in range(self.obs.shape[1]):
            self.store_data.append([])
            self.store_data_std.append([])
            self.store_syns_n_reds.append([])
            print(self.scalable, 'scalable in larger func')
            func = int_alg_per_varb(self.obs, self.acs, chosen_varb, self.mines, self.mines_std,
                                     self.num_iterations, scalable=self.scalable)
            func.run_varb()
            mi = self.mines[chosen_varb][-1]
            mi_std = self.mines_std[chosen_varb][-1]
            self.store_data[-1].append(mi)
            self.store_data[-1].append(func.net_synergy)
            self.store_data[-1].append(func.redundant_reductions)
            self.store_data_std[-1].append(mi_std)
            self.store_data_std[-1].append(func.syn_std)
            self.store_data_std[-1].append(func.redundant_reductions_std)
            print(f'Here are the values for variable {chosen_varb}s information, synergistic information and redundancy respectively {self.store_data[-1]}')
            self.store_syns_n_reds[-1].append(func.synergistic)
            self.store_syns_n_reds[-1].append(func.redundants)
            print(f'Here are the variables that have redundant info and combine synergistically wrt variable {chosen_varb}, :{self.store_syns_n_reds[-1]}')
        return self.store_data, self.store_data_std, self.store_syns_n_reds


class int_alg_mnist:
    def __init__(self, obs, acs, num_iterations, chosen_varb, name, scalable=True):
        self.num_iterations = num_iterations
        self.chosen_varb = chosen_varb
        self.name = name
        self.mi_method = "mine_fa" if scalable else "mine"
        self.obs = np.nan_to_num(obs).round(decimals=2)
        self.acs = np.nan_to_num(acs).round(decimals=2)
        self.mines = []
        self.mines_std = []
        self.expts = 5

    def do_mine(self, data, acs, num_iterations):
        if self.mi_method == "mine":
            return mine(data, acs, num_iterations).run()
        else:
            return mine_fa(data, acs, num_iterations).run()

    def get_mis(self):
        print('Getting MI')
        i = self.chosen_varb
        results = self.do_mine(self.obs[:, i], self.acs, self.num_iterations)
        means = results.mean(axis=0)
        std = results.std(axis=0)
        self.mines.append(means)
        self.mines_std.append(std)
        print(f'MI for varb {i} = {means[-1]} ± {std[-1]*2/math.sqrt(self.expts)}')

    def run(self):
        self.get_mis()
        chosen_varb = self.chosen_varb
        func = int_alg_per_varb_mnist(self.obs, self.acs, chosen_varb, self.mines, self.mines_std, self.num_iterations)
        func.run_varb()
        mi = self.mines[0][-1]  # Since mines only contains one element now
        mi_std = self.mines_std[0][-1]
        data = [mi, func.net_synergy, func.redundant_reductions]  # Note: func.redundant_reductions may be a list
        data_std = [mi_std, func.syn_std, func.redundant_reductions_std]
        syns_n_reds = [func.synergistic, func.redundants]
        print(f'Values for variable {chosen_varb}: Information={data[0]}, Synergistic Information={data[1]}, Redundancy={data[2]}')
        print(f'Redundant and synergistic variables wrt variable {chosen_varb}: {syns_n_reds}')

        # Determine the shapes
        num_vars = self.obs.shape[1]
        data_shape = (num_vars, 3)  # [MI, Net Synergy, Redundancy]
        syns_n_reds_shape = (num_vars, 2)  # [Synergistic variables, Redundant variables]

        # File paths
        data_file = f'interpretability_{self.name}{self.num_iterations}.pickle'
        data_std_file = f'interpretability_std_{self.name}{self.num_iterations}.pickle'
        syns_n_reds_file = f'syns_and_reds_{self.name}{self.num_iterations}.pickle'

        # Load or initialize data arrays
        if os.path.exists(data_file):
            data_array = np.load(data_file, allow_pickle=True)
        else:
            data_array = np.empty(data_shape, dtype=object)
        if os.path.exists(data_std_file):
            data_std_array = np.load(data_std_file, allow_pickle=True)
        else:
            data_std_array = np.empty(data_shape, dtype=object)
        if os.path.exists(syns_n_reds_file):
            syns_n_reds_array = np.load(syns_n_reds_file, allow_pickle=True)
        else:
            syns_n_reds_array = np.empty(syns_n_reds_shape, dtype=object)
            syns_n_reds_array[:, 0] = [[] for _ in range(num_vars)]
            syns_n_reds_array[:, 1] = [[] for _ in range(num_vars)]

        # Update the arrays at the correct index
        data_array[chosen_varb, :] = data
        data_std_array[chosen_varb, :] = data_std
        syns_n_reds_array[chosen_varb, 0] = func.synergistic
        syns_n_reds_array[chosen_varb, 1] = func.redundants

        # Save the updated "arrays" back to the files
        with open(data_file, 'wb') as h:
            pickle.dump(data_array, h)
        with open(data_std_file, 'wb') as h:
            pickle.dump(data_std_array, h)
        with open(syns_n_reds_file, 'wb') as h:
            pickle.dump(syns_n_reds_array, h)
            
        # np.save(data_file, data_array)
        # np.save(data_std_file, data_std_array)
        # np.save(syns_n_reds_file, syns_n_reds_array)

