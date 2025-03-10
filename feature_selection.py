import numpy as np
import torch
import torch.nn as nn
import scipy.stats as stats
import torch.optim as optim
import gc
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import make_scorer,  r2_score
from synth_gen import generate_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class pi_alg:
    def __init__(self, obs, acs):
        """
        Initializes the PIAlg with a suitable model and datasets (observations and actions)
        based on whether the target data is continuous or discrete.
        """
        self.obs = obs
        self.acs = acs
        if self.is_continuous(acs):
            self.model = LinearRegression()
            self.scoring = make_scorer(r2_score)
        else:
            self.model = LogisticRegression()
            self.scoring = 'accuracy'

    def is_continuous(self, data):
        return np.unique(data).size > 10 or not np.issubdtype(data.dtype, np.integer)

    def fit(self):
        self.model.fit(self.obs, self.acs)

    def compute_importance(self, n_repeats=30, random_state=None):
        result = permutation_importance(self.model, self.obs, self.acs, n_repeats=n_repeats,
                                        random_state=random_state, scoring=self.scoring)
        return result.importances_mean

    def run(self, threshold=0.01):
        self.fit()
        importances = self.compute_importance()
        return np.where(importances > threshold)[0]
    
class wollstadt_alg:
    def __init__(self, features, targets, num_iterations):
        self.num_iterations = num_iterations
        self.features = features.round(decimals =2)
        self.targets = targets.round(decimals =2)
        #print(self.features, self.targets)
        self.max_iter = num_iterations

    def fit_mlp(self, features, targets, mlp_regressor,  feature_mask=None, max_iter=None, original_loss=None ):
        if feature_mask is not None:
            features = features * feature_mask
            features[features == 0] = 0.0
        criterion = nn.MSELoss()
        optimizer = optim.Adam(mlp_regressor.parameters(), lr=0.001)
        
        if max_iter == None:
            max_iter = self.max_iter

        #print(features, targets)
        for epoch in range(max_iter):
            predictions = mlp_regressor(features)
            
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        
        return loss.item()
    
    def evaluate_feature_importance(self, features, targets, ci):
        print(targets.shape, 'targets')
        print(features.shape, 'features')
        targets = torch.tensor(targets, dtype=torch.float32, device=device)
        features = torch.tensor(features, dtype=torch.float32, device=device)
        targets = self.normalize_torch(targets)
        features = self.normalize_torch(features)
        feature_mask = torch.zeros(features.shape[-1], device=device)
        mlp_regressors = self.create_mlp_regressors(features.shape[-1], targets.shape[-1], count=3)
        original_losses = [self.fit_mlp(torch.rand(features.shape), targets, regressor, torch.ones(features.shape[-1], device=device), max_iter=self.max_iter) for regressor in mlp_regressors]
        original_mean_loss, original_conf_interval = self.calculate_confidence_interval(original_losses, ci)
        print(original_losses, 'ol')
        important_features = []
        important_features, feature_mask, done = self.find_important_features(important_features, features, targets, mlp_regressors, feature_mask, original_conf_interval, ci)
        while not done:
            mlp_regressors = self.create_mlp_regressors(features.shape[-1], targets.shape[-1], count=3)
            original_losses = [self.fit_mlp(features, targets, regressor, feature_mask, max_iter=self.max_iter) for regressor in mlp_regressors]
            original_mean_loss, original_conf_interval = self.calculate_confidence_interval(original_losses, ci)
            important_features, feature_mask, done = self.find_important_features(important_features, features, targets, mlp_regressors, feature_mask, original_conf_interval, ci)

        del features, targets, mlp_regressors, feature_mask, original_conf_interval
        gc.collect()
        return important_features

    
    def normalize_torch(self, data):
        epsilon = 1e-8
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        return (data - mean) / (std + epsilon)


    def create_mlp_regressors(self, input_size, output_size, count=3):
        return [nn.Sequential(
            nn.Linear(input_size, 50), nn.ReLU(),nn.Linear(50,50), nn.ReLU(),nn.Linear(50,50),nn.ReLU(), nn.Linear(50, output_size)
        ).to(device) for _ in range(count)]

    
    def calculate_confidence_interval(self, losses, ci):
        mean_loss = np.mean(losses)
        sem = stats.sem(losses) if np.std(losses) > 0 else 0
        conf_interval = stats.t.interval(ci, len(losses)-1, loc=mean_loss, scale=sem)
        conf_interval = tuple(mean_loss if np.isnan(val) else val for val in conf_interval)
        return mean_loss, conf_interval

    def find_important_features(self, important_features, features, targets, regressors, feature_mask, original_conf_interval, ci):
        means, cis = [], []
        for column in range(features.shape[1]):
            if feature_mask[column] == 0:
                print(f'checking column {column} as not already added')
                temp_mask = feature_mask.clone()
                temp_mask[column] = 1
                regressors = self.create_mlp_regressors(features.shape[-1], targets.shape[-1], count=3)
                modified_losses = [self.fit_mlp(features, targets, regressor, temp_mask, original_loss=original_conf_interval[1]) for regressor in regressors]
                mean, modified_conf_interval = self.calculate_confidence_interval(modified_losses, ci)
                means.append(mean)
                cis.append(modified_conf_interval)
            else:
                print(f'not checking column {column} as already added')
                means.append(10)
                cis.append(10)
        print(f'mean losses are: {means}')
        selected_feature = means.index(min(means))
        selected_conf_interval = cis[selected_feature]
        print('selected feature is', {selected_feature})
        print(original_conf_interval, selected_conf_interval, 'ci')
        if not isinstance(selected_conf_interval, int) and selected_conf_interval[1] < original_conf_interval[0]:
                feature_mask[selected_feature] = 1 # Mark feature as unimportant
                important_features.append(selected_feature)
                print(f'feature {selected_feature} is being added.')
                done = False
        else:
                print(f'no more features being added as they do not improve learning.')
                done = True
        return important_features, feature_mask, done
    
    def run(self):
        important_features = self.evaluate_feature_importance(self.features, self.targets, 0.95)
        return important_features
    

class terc_alg:
    def __init__(self, features, targets, num_iterations):
        self.num_iterations = num_iterations
        self.features = features.round(decimals =2)
        self.targets = targets.round(decimals =2)
        self.max_iter = num_iterations

    def fit_mlp(self, features, targets, mlp_regressor,  feature_mask=None, max_iter=None, original_loss=None ):
        if feature_mask is not None:
            features = features * feature_mask
        criterion = nn.MSELoss()
        optimizer = optim.Adam(mlp_regressor.parameters(), lr=0.001)
        
        if max_iter == None:
            max_iter = self.max_iter

        for epoch in range(max_iter):
            predictions = mlp_regressor(features)
            
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if original_loss is not None and loss.item() < original_loss:
                break

        
        return loss.item()
    
    def evaluate_feature_importance(self, features, targets, ci):
        print(targets.shape, 'targets')
        print(features.shape, 'features')
        targets = torch.tensor(targets, dtype=torch.float32, device=device)
        features = torch.tensor(features, dtype=torch.float32, device=device)
        targets = self.normalize_torch(targets)
        features = self.normalize_torch(features)
        feature_mask = torch.ones(features.shape[-1], device=device)
        mlp_regressors = self.create_mlp_regressors(features.shape[-1], targets.shape[-1], count=3)
        original_losses = [self.fit_mlp(features, targets, regressor, feature_mask, max_iter=self.max_iter) for regressor in mlp_regressors]
        original_mean_loss, original_conf_interval = self.calculate_confidence_interval(original_losses, ci)
        unimportant_features = []
        unimportant_features, feature_mask = self.find_unimportant_features(unimportant_features, features, targets, mlp_regressors, feature_mask, original_conf_interval, ci)
        del features, targets, mlp_regressors, feature_mask, original_conf_interval
        gc.collect()
        return unimportant_features

    
    def normalize_torch(self, data):
        epsilon = 1e-8
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        return (data - mean) / (std + epsilon)


    def create_mlp_regressors(self, input_size, output_size, count=3):
        return [nn.Sequential(
            nn.Linear(input_size, 50), nn.ReLU(), nn.Linear(50, output_size)
        ).to(device) for _ in range(count)]

    
    def calculate_confidence_interval(self, losses, ci):
        mean_loss = np.mean(losses)
        sem = stats.sem(losses) if np.std(losses) > 0 else 0
        conf_interval = stats.t.interval(ci, len(losses)-1, loc=mean_loss, scale=sem)
        conf_interval = tuple(mean_loss if np.isnan(val) else val for val in conf_interval)
        return mean_loss, conf_interval

    def find_unimportant_features(self, unimportant_features, features, targets, regressors, feature_mask, original_conf_interval, ci):
        self.important_features = []
        for column in range(features.shape[1] - 1, - 1, -1):
            temp_mask = feature_mask.clone()
            temp_mask[column] = 0
            regressors = self.create_mlp_regressors(features.shape[-1], targets.shape[-1], count=3)
            modified_losses = [self.fit_mlp(features, targets, regressor, temp_mask, original_loss=original_conf_interval[1]) for regressor in regressors]
            _, modified_conf_interval = self.calculate_confidence_interval(modified_losses, ci)
            if modified_conf_interval[0] > original_conf_interval[1] and feature_mask[column] == 1:
                self.important_features.append(column)
                print(f'feature {column} is being kept as {modified_conf_interval[0]} > {original_conf_interval[1]}')
            else:
                feature_mask[column] = 0  # Mark feature as unimportant
                unimportant_features.append(column)
                print(f'feature {column} is being removed as {modified_conf_interval[0]} +< {original_conf_interval[1]}')
            
        return unimportant_features, feature_mask
    
    def run(self):
        unimportant_features = self.evaluate_feature_importance(self.features, self.targets, 0.95)
        return self.important_features

class pidf_alg:
    def __init__(self, features, targets, num_iterations):
        self.num_iterations = num_iterations
        self.features = features.round(decimals =2)
        self.targets = targets.round(decimals =2)
        self.targets = torch.tensor(self.targets, dtype=torch.float32, device=device)
        self.features = torch.tensor(self.features, dtype=torch.float32, device=device)
        self.targets = self.normalize_torch(self.targets)
        self.features = self.normalize_torch(self.features)
        self.max_iter = num_iterations
        self.ci = 0.95

    def fit_mlp(self, features, targets, mlp_regressor,  feature_mask=None, max_iter=None, original_loss=None ):
        if feature_mask is not None:
            features = features * feature_mask
        criterion = nn.MSELoss()
        optimizer = optim.Adam(mlp_regressor.parameters(), lr=0.001)
        if max_iter == None:
            max_iter = self.max_iter
        for epoch in range(max_iter):
            predictions = mlp_regressor(features)
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if original_loss is not None and loss.item() < original_loss:
                break

        
        return loss.item()
    
    def get_foi_mi(self, features, targets, FOI_index):
        temp_mask = torch.zeros(features.shape[-1], device=device)
        regressors = self.create_mlp_regressors(features.shape[-1], targets.shape[-1], count=3)
        losses_without = [self.fit_mlp(features, targets, regressor, temp_mask) for regressor in regressors]
        temp_mask[FOI_index] = 1
        regressors = self.create_mlp_regressors(features.shape[-1], targets.shape[-1], count=3)
        losses_with = [self.fit_mlp(features, targets, regressor, temp_mask) for regressor in regressors]
        return np.array(losses_without) - np.array(losses_with)
        
    
    def get_mis(self, features, FOI_index):
        mis, mi_indexes = [], []
        foi_mask = torch.zeros(features.shape[-1], device=device)
        foi_mask[FOI_index] = 1
        FOI_vals = features*foi_mask
        red_fnoi = []
        feature_mask = torch.zeros(features.shape[-1], device=device)
        for column in range(features.shape[1]):
            if column != FOI_index:
                temp_mask = feature_mask.clone()
                temp_mask[column] = 1
                regressors = self.create_mlp_regressors(features.shape[-1], FOI_vals.shape[-1], count=3)
                modified_losses = [self.fit_mlp(features, FOI_vals, regressor, temp_mask) for regressor in regressors]
                temp_mask[column] = 0
                modified_losses_none = [self.fit_mlp(features, FOI_vals, regressor, temp_mask) for regressor in regressors]
                mean_mi, modified_conf_interval = self.calculate_confidence_interval(np.array(modified_losses_none) - np.array(modified_losses), self.ci)
                if modified_conf_interval[0] > 0.005:
                    red_fnoi.append(column)
                mis.append(mean_mi)
                mi_indexes.append(column)

        print(mis, mi_indexes)
        mis, mi_indexes = zip(*sorted(zip(mis, mi_indexes)))
        return mi_indexes[::-1], red_fnoi
    
    def adding_foi(self, features, target, FOI_index):
        temp_mask = torch.ones(features.shape[-1], device=device)
        temp_mask[FOI_index] = 0
        regressors = self.create_mlp_regressors(features.shape[-1], target.shape[-1], count=3)
        losses_without = [self.fit_mlp(features, target, regressor, temp_mask) for regressor in regressors]
        regressors = self.create_mlp_regressors(features.shape[-1], target.shape[-1], count=3)
        losses_with = [self.fit_mlp(features, target, regressor) for regressor in regressors]
        return np.array(losses_without) - np.array(losses_with)
    
    def check_theta(self, features, target, FOI_index, FNOI_index):
        lht_with_fnoi = self.adding_foi(features, target, FOI_index)
        temp_mask = torch.ones(features.shape[-1], device=device)
        temp_mask[FNOI_index] = 0
        features = features * temp_mask
        rht_without_fnoi = self.adding_foi(features, target, FOI_index)
        theta_means = lht_with_fnoi - rht_without_fnoi
        return theta_means
    
    def PIDF_per_foi(self, features, targets, FOI_index):
        foi_mi = self.get_foi_mi(features, targets, FOI_index)
        features_mask = torch.ones(features.shape[-1], device=device)
        redundant_contribution, synergistic_fnoi = [], []
        FNOI_indexes, redundnat_fnoi = self.get_mis(features, FOI_index)
        for FNOI_index in FNOI_indexes:
            theta_vals = self.check_theta(features, targets, FOI_index, FNOI_index)
            theta_mean, theta_ci = self.calculate_confidence_interval(theta_vals, self.ci)
            if theta_ci[1] < 0:
                features_mask[FNOI_index] = 0
                features = features * features_mask
                redundant_contribution.append(-theta_vals)
            elif theta_ci[0] > 0:
                synergistic_fnoi.append(FNOI_index)
            else:
                features_mask[FNOI_index] = 0
                features = features * features_mask

        total_syn = self.adding_foi(features, targets, FOI_index)-foi_mi

        return foi_mi, total_syn, redundant_contribution, synergistic_fnoi, redundnat_fnoi

    
    def do_pidf(self):
        selected_feats = []
        checkers = []
        all_red_fnoi = []
        mean_tis = []
        for FOI_index in range(self.features.shape[-1]):
            mi, synergy, redundant_contributions, synergistic_fnoi, redundant_fnoi = self.PIDF_per_foi(self.features, self.targets, FOI_index)
            all_red_fnoi.append(redundant_fnoi)
            mean_ti, ti_ci = self.calculate_confidence_interval(np.array(mi)+np.array(synergy), self.ci) if synergy != [] else self.calculate_confidence_interval(np.array(mi), self.ci) 
            _, red_ci = self.calculate_confidence_interval(np.sum([np.array(red) for red in redundant_contributions], axis=0), self.ci) if redundant_contributions else (0, (0, 0))
            if ti_ci[0] > red_ci[1]:
                print(f'FOI {FOI_index} has been included due to its information contribution')
                selected_feats.append(FOI_index)
            elif ti_ci[1] < red_ci[0]:
                print(f'FOI {FOI_index} not included as redundant')
            elif ti_ci[0] < 0:
                print(f'FOI {FOI_index} not included as irrelevant')
            else:
                print(f'FOI {FOI_index} needs to be checked')
                checkers.append(FOI_index)
                print(checkers)
                mean_tis.append(mean_ti)

        sorted_indices = np.argsort(mean_tis)[::-1]
        sorted_checkers = [checkers[i] for i in sorted_indices]
        checkers = sorted_checkers
        for checker in checkers:
            print(checker)
            if not set(all_red_fnoi[checker]).intersection(set(selected_feats)):
                print(f'checker {checker} added as it was perfectly redundant')
                selected_feats.append(checker)
        
        return selected_feats

    
    def normalize_torch(self, data):
        epsilon = 1e-8
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        return (data - mean) / (std + epsilon)


    def create_mlp_regressors(self, input_size, output_size, count=3):
        return [nn.Sequential(
            nn.Linear(input_size, 50), nn.ReLU(),nn.Linear(50,50),nn.ReLU(), nn.Linear(50, output_size)
        ).to(device) for _ in range(count)]

    
    def calculate_confidence_interval(self, losses, ci):
        mean_loss = np.mean(losses)
        sem = stats.sem(losses) if np.std(losses) > 0 else 0
        conf_interval = stats.t.interval(ci, len(losses)-1, loc=mean_loss, scale=sem)
        conf_interval = tuple(mean_loss if np.isnan(val) else val for val in conf_interval)
        return mean_loss, conf_interval

def run_feature_selection(names=['RVQ', 'SVQ', 'MSP', 'WT', 'TERC1', 'TERC2', 'UBR', 'SG'], num_iters=500):    
    for name in names:
        # Generate features and targets for the given dataset name
        feats, targs = generate_data(name)
        
        # PIDF algorithm
        important_feats = pidf_alg(feats, targs, num_iters).do_pidf()
        print(important_feats)
        np.save(f'PIDF_{name}.npy', np.array(important_feats))
        print(f'PIDF alg selected the following features as important: {important_feats} for the {name} dataset')
        
        # Wollstadt algorithm
        important_feats = wollstadt_alg(feats, targs, num_iters).run()
        np.save(f'Wollstadt_{name}.npy', np.array(important_feats))
        print(f'Wollstadt alg selected the following features as important: {important_feats} for the {name} dataset')
        
        # TERC algorithm
        important_feats = terc_alg(feats, targs, num_iters).run()
        np.save(f'TERC_{name}.npy', np.array(important_feats))
        print(f'TERC alg selected the following features as important: {important_feats} for the {name} dataset')
        
        # PI algorithm
        important_feats = pi_alg(feats, targs).run()
        np.save(f'PI_{name}.npy', np.array(important_feats))
        print(f'PI alg selected the following features as important: {important_feats} for the {name} dataset')
