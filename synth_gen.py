import random
import numpy as np
import pandas as pd
from sklearn import datasets as dsets
import math
import numpy.random as ran
import numpy as np
import numpy as np
import torch
from torchvision import datasets, transforms
import re


def to_numeric(data):
    for i in range(data.shape[1]):
        if data.iloc[:, i].dtype != 'int64' and data.iloc[:, i].dtype != 'float64':
            data.iloc[:, i], unique_values = pd.factorize(data.iloc[:, i])
    return data.astype(float)

def generate_data(nme, n_points=1000):
    obs = []
    acs = []
    if nme == 'RVQ':
        for _ in range(n_points):
            initial_sublist = [random.randint(0,1) for _ in range(2)]
            obs.append(initial_sublist + [initial_sublist[1]]) 
            acs.append([initial_sublist[0]+2*initial_sublist[1]])
        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
    elif nme == 'SVQ':
        for _ in range(n_points):
            initial_sublist = [random.randint(0,1) for _ in range(2)]
            obs.append(initial_sublist ) 
            if initial_sublist[0] == initial_sublist[1]:
                acs.append([0])
            else:
                acs.append([1])
        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
    elif nme == 'MSP':
        for _ in range(n_points):
            initial_sublist = [random.randint(0,1) for _ in range(2)]
            obs.append(initial_sublist + [sum(initial_sublist)]) 
            acs.append([sum(initial_sublist)])
        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
    elif nme == 'WT':
        for _ in range(n_points):
            ep = ran.normal(loc=0.0, scale=1.0, size=None)
            ep2 = ran.normal(loc=0.0, scale=1.0, size=None)
            gn = ran.normal(loc=0.0, scale=1.0, size=None)
            gn2 = ran.normal(loc=0.0, scale=1.0, size=None)
            y = math.sin(ep) + gn*0.1
            X_1 = (ep) + gn2*0.1
            X_2 = 0.8*ep + 0.2*ep2 +0.01*gn2
            obs.append([X_1,X_2,gn2] ) 
            acs.append([y])
        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
    elif nme == 'TERC1':
        for _ in range(n_points):
            initial_sublist = [random.randint(0,1) for _ in range(3)]
            obs.append(initial_sublist + [initial_sublist[0]] * 3) 
            if initial_sublist[0] == initial_sublist[1] == initial_sublist[2]:
                acs.append([1])
            elif initial_sublist[0] != initial_sublist[1] != initial_sublist[2]:
                acs.append([1])
            else:
                acs.append([0])
        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
    elif nme == 'TERC2':
        for _ in range(n_points):
            initial_sublist = [random.randint(0,1) for _ in range(3)]
            obs.append(initial_sublist + initial_sublist) 
            if initial_sublist[0] == initial_sublist[1] == initial_sublist[2]:
                acs.append([1])
            else:
                acs.append([0])
        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
    elif nme == 'UBR':
        for _ in range(n_points):
            X_0 = random.normalvariate(0,1)
            S = random.normalvariate(0,1)
            delta = random.uniform(-1, 1)
            epsilon = random.uniform(-0.5, 0.5)
            
            theta =  np.exp(-random.randint(0, 10))
            X_1 = 3*X_0 + delta
            X_2 = S + X_1
            y = S+epsilon
            X_3 = y+ theta
            obs.append([X_0,X_1,X_2,X_3] ) 
            acs.append([y])
        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
    elif nme == 'SG':
        acs = []
        obs = []

        for _ in range(n_points):
            x = random.randint(0, 1)
            acs.append([x])

            # Generate sublist based on the value of x
            sublist = []
            threshold = 0.95 if x == 0 else 0.05
            if random.uniform(0, 1) > threshold:
                j = random.randint(0, 2)
                sublist = [[0,0],[0,1],[1,0]][j]
            else:
                sublist = [1, 1]

            sublist.append(random.uniform(0, 1) > 0.8 if x == 0 else random.uniform(0, 1) <= 0.8)

            obs.append(sublist)

    elif nme == 'boston':
        boston_dataset = dsets.load_boston()
        obs = np.array(boston_dataset.data)
        acs = np.array(boston_dataset.target)
        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'iris':
        boston_dataset = dsets.load_iris()
        obs = np.array(boston_dataset.data)
        acs = np.array(boston_dataset.target)
        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'whitewine':
        data = pd.read_csv("winequality-white.csv", delimiter=";")
        data = data.fillna(0)
        data=(data-data.min())/(data.max()-data.min())
        data = data.to_numpy()
        obs = np.array(data[:, :-1])  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(data[:, -1])
        print(obs)
        print(acs)
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'whitewine_duplicate':
        data = pd.read_csv("winequality-white.csv", delimiter=";")
        data = data.fillna(0)
        data=(data-data.min())/(data.max()-data.min())
        data = data.to_numpy()
        obs = np.array(data[:, :-1])  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(data[:, -1])
        duplicated_feature = obs[:, [7]]  # Extracting the first column and keeping its 2D shape
        obs = np.hstack((obs, duplicated_feature))  # Appending the duplicated feature to 'obs'
        print(obs)
        print(acs)
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'abalone':
        data = to_numeric(pd.read_csv(f"{nme}_orig.csv", delimiter=","))
        data = data.fillna(0)
        data=(data-data.min())/(data.max()-data.min())
        data = data.to_numpy()
        obs = np.array(data[:, :-1])  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(data[:, -1])
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'abalone_duplicate':
        data = to_numeric(pd.read_csv(f"abalone_orig.csv", delimiter=","))
        data = data.fillna(0)
        data=(data-data.min())/(data.max()-data.min())
        data = data.to_numpy()
        obs = np.array(data[:, :-1])  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(data[:, -1])
        duplicated_feature = obs[:, [2]]  # Extracting the first column and keeping its 2D shape
        obs = np.hstack((obs, duplicated_feature))
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'housing':
        data = to_numeric(pd.read_csv(f"{nme}.csv", delimiter=","))
        print(data.columns)
        indexes = list(range(data.shape[1]))
        del indexes[-2]
        data = data.fillna(0)
        data=(data-data.min())/(data.max()-data.min())
        data = np.array(data)
        obs = np.array(data[:, indexes])  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(data[:, -2])
        print(obs)
        print(acs)
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'housing_duplicate':
        data = to_numeric(pd.read_csv(f"housing.csv", delimiter=","))
        print(data.columns)
        indexes = list(range(data.shape[1]))
        del indexes[-2]
        data = data.fillna(0)
        data=(data-data.min())/(data.max()-data.min())
        data = np.array(data)
        obs = np.array(data[:, indexes])  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(data[:, -2])
        duplicated_feature = obs[:, [0]]  # Extracting the first column and keeping its 2D shape
        obs = np.hstack((obs, duplicated_feature))  # Appending the duplicated feature to 'obs'
        print(obs)
        print(acs)
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'adult':
        data = to_numeric(pd.read_csv(f"{nme}.csv", delimiter=","))
        data = data.fillna(0)
        data=(data-data.min())/(data.max()-data.min())
        data = np.array(data)
        obs = np.array(data[:, :-1])  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(data[:, -1])
        print(obs)
        print(acs)
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'cleveland':
        data = to_numeric(pd.read_csv(f"processed.cleveland.data", delimiter=","))
        data = data.fillna(0)
        data=(data-data.min())/(data.max()-data.min())
        data = np.array(data)
        obs = np.array(data[:, :-1])  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(data[:, -1])
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'thyroid':
        data = to_numeric(pd.read_csv(f"thyroid0387.data", delimiter=","))
        print(data.columns)
        data = data.fillna(0)
        data=(data-data.min())/(data.max()-data.min())
        print(data.dtypes)
        print(data)
        data = np.array(data)
        obs = np.array(data[:, 1:])  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(data[:, 0])
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'wdbc':
        data = to_numeric(pd.read_csv(f"wdbc.data", delimiter=","))
        print(data.columns)
        data = data.fillna(0)
        data=(data-data.min())/(data.max()-data.min())
        data = np.array(data)
        
        obs = np.array(data[:, 2:])  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(data[:, 1])
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'bupa':
        data = to_numeric(pd.read_csv(f"bupa.data", delimiter=","))
        print(data.columns)
        data = data.fillna(0)
        data=(data-data.min())/(data.max()-data.min())
        data = np.array(data)
        
        obs = np.array(data[:, :-2])  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(data[:, -2])
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'brca':
        data = to_numeric(pd.read_csv(f"BRCA.csv", delimiter=","))
        print(data.columns)
        print(data.shape)
        data = data.fillna(0)
        data=(data-data.min())/(data.max()-data.min())
        data = np.array(data)
        
        obs = np.array(data[:, 1:-1])  # shape: (batch_size, sequence_len, input_dim=3)
        print(obs)
        acs = np.array(data[:, -1])
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'brca_small':
        data = to_numeric(pd.read_csv(f"BRCA.csv", delimiter=","))
        print(data.columns)
        print(data.shape)
        cancer_colums = ['BCL11A','BRCA2', 'CCND1','TEX14','BRCA1', 'EZH2','SLC22A5','CDK6','LFNG','IGF1R','BRCA_Subtype_PAM50']
        data = data[cancer_colums].fillna(0)
        data=(data-data.min())/(data.max()-data.min())
        data = np.array(data)
        
        obs = np.array(data[:, :-1])  # shape: (batch_size, sequence_len, input_dim=3)
        print(obs.shape)
        acs = np.array(data[:, -1])
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    
    elif nme == 'timme_neurons_day4':
        # 4, 7, 12, 16, 20, 25, 31, and 33
        obs = pd.read_csv("2-2-4.spk.txt", delimiter=" ")
        processed_obs = process_neurons(obs)
        
        # Randomly select any 10 columns available from the processed DataFrame
        selected_columns = random.sample(list(processed_obs.columns), 10)
        obs = processed_obs[selected_columns]
        
        # Sample rows to have a fraction (here 1000 rows, with replacement if needed)
        obs = obs.sample(frac=(1), replace=True, random_state=1)

        # Find a column index (for demonstration, this picks a random index in the range 0 to 9)
        max_cumsum_index = random.randint(0, 9)
        print(max_cumsum_index)
        
        # Shift the chosen column downwards by one position
        acs = obs.iloc[:, max_cumsum_index].shift(1)
        # Drop the chosen column using its label
        obs = obs.drop(obs.columns[max_cumsum_index], axis=1)

        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'timme_neurons_day7':
        # 4, 7, 12, 16, 20, 25, 31, and 33
        obs = pd.read_csv("2-2-7.spk.txt", delimiter=" ")
        processed_obs = process_neurons(obs)
        
        # Randomly select any 10 columns available from the processed DataFrame
        selected_columns = random.sample(list(processed_obs.columns), 10)
        obs = processed_obs[selected_columns]
        
        # Sample rows to have a fraction (here 1000 rows, with replacement if needed)
        obs = obs.sample(frac=(1), replace=True, random_state=1)

        # Find a column index (for demonstration, this picks a random index in the range 0 to 9)
        max_cumsum_index = random.randint(0, 9)
        print(max_cumsum_index)
        
        # Shift the chosen column downwards by one position
        acs = obs.iloc[:, max_cumsum_index].shift(1)
        # Drop the chosen column using its label
        obs = obs.drop(obs.columns[max_cumsum_index], axis=1)

        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'timme_neurons_day12':
        # 4, 7, 12, 16, 20, 25, 31, and 33
        obs = pd.read_csv("2-2-12.spk.txt", delimiter=" ")
        processed_obs = process_neurons(obs)
        
        # Randomly select any 10 columns available from the processed DataFrame
        selected_columns = random.sample(list(processed_obs.columns), 10)
        obs = processed_obs[selected_columns]
        
        # Sample rows to have a fraction (here 1000 rows, with replacement if needed)
        obs = obs.sample(frac=(1), replace=True, random_state=1)

        # Find a column index (for demonstration, this picks a random index in the range 0 to 9)
        max_cumsum_index = random.randint(0, 9)
        print(max_cumsum_index)
        
        # Shift the chosen column downwards by one position
        acs = obs.iloc[:, max_cumsum_index].shift(1)
        # Drop the chosen column using its label
        obs = obs.drop(obs.columns[max_cumsum_index], axis=1)

        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'timme_neurons_day16':
        # 4, 7, 12, 16, 20, 25, 31, and 33
        obs = pd.read_csv("2-2-16.spk.txt", delimiter=" ")
        processed_obs = process_neurons(obs)
        
        # Randomly select any 10 columns available from the processed DataFrame
        selected_columns = random.sample(list(processed_obs.columns), 10)
        obs = processed_obs[selected_columns]
        
        # Sample rows to have a fraction (here 1000 rows, with replacement if needed)
        obs = obs.sample(frac=(1), replace=True, random_state=1)

        # Find a column index (for demonstration, this picks a random index in the range 0 to 9)
        max_cumsum_index = random.randint(0, 9)
        print(max_cumsum_index)
        
        # Shift the chosen column downwards by one position
        acs = obs.iloc[:, max_cumsum_index].shift(1)
        # Drop the chosen column using its label
        obs = obs.drop(obs.columns[max_cumsum_index], axis=1)

        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'timme_neurons_day20':
        # 4, 7, 12, 16, 20, 25, 31, and 33
        obs = pd.read_csv("2-2-20.spk.txt", delimiter=" ")
        processed_obs = process_neurons(obs)
        
        # Randomly select any 10 columns available from the processed DataFrame
        selected_columns = random.sample(list(processed_obs.columns), 10)
        obs = processed_obs[selected_columns]
        
        # Sample rows to have a fraction (here 1000 rows, with replacement if needed)
        obs = obs.sample(frac=(1), replace=True, random_state=1)

        # Find a column index (for demonstration, this picks a random index in the range 0 to 9)
        max_cumsum_index = random.randint(0, 9)
        print(max_cumsum_index)
        
        # Shift the chosen column downwards by one position
        acs = obs.iloc[:, max_cumsum_index].shift(1)
        # Drop the chosen column using its label
        obs = obs.drop(obs.columns[max_cumsum_index], axis=1)

        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'timme_neurons_day25':
        # 4, 7, 12, 16, 20, 25, 31, and 33
        obs = pd.read_csv("2-2-25.spk.txt", delimiter=" ")
        processed_obs = process_neurons(obs)
        
        # Randomly select any 10 columns available from the processed DataFrame
        selected_columns = random.sample(list(processed_obs.columns), 10)
        obs = processed_obs[selected_columns]
        
        # Sample rows to have a fraction (here 1000 rows, with replacement if needed)
        obs = obs.sample(frac=(1), replace=True, random_state=1)

        # Find a column index (for demonstration, this picks a random index in the range 0 to 9)
        max_cumsum_index = random.randint(0, 9)
        print(max_cumsum_index)
        
        # Shift the chosen column downwards by one position
        acs = obs.iloc[:, max_cumsum_index].shift(1)
        # Drop the chosen column using its label
        obs = obs.drop(obs.columns[max_cumsum_index], axis=1)

        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'timme_neurons_day31':
        # 4, 7, 12, 16, 20, 25, 31, and 33
        obs = pd.read_csv("2-2-31.spk.txt", delimiter=" ")
        processed_obs = process_neurons(obs)
        
        # Randomly select any 10 columns available from the processed DataFrame
        selected_columns = random.sample(list(processed_obs.columns), 10)
        obs = processed_obs[selected_columns]
        
        # Sample rows to have a fraction (here 1000 rows, with replacement if needed)
        obs = obs.sample(frac=(1), replace=True, random_state=1)

        # Find a column index (for demonstration, this picks a random index in the range 0 to 9)
        max_cumsum_index = random.randint(0, 9)
        print(max_cumsum_index)
        
        # Shift the chosen column downwards by one position
        acs = obs.iloc[:, max_cumsum_index].shift(1)
        # Drop the chosen column using its label
        obs = obs.drop(obs.columns[max_cumsum_index], axis=1)

        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    elif nme == 'timme_neurons_day33':
        # 4, 7, 12, 16, 20, 25, 31, and 33
        obs = pd.read_csv("2-2-33.spk.txt", delimiter=" ")
        
        # Randomly select any 10 columns available from the processed DataFrame
        selected_columns = random.sample(list(processed_obs.columns), 10)
        obs = processed_obs[selected_columns]
        
        # Sample rows to have a fraction (here 1000 rows, with replacement if needed)
        obs = obs.sample(frac=(1), replace=True, random_state=1)
        cumulative_sums = obs.cumsum().iloc[-1]

        # Find a column index (for demonstration, this picks a random index in the range 0 to 9)
        max_cumsum_index = random.randint(0, 9)
        print(max_cumsum_index)
        
        # Shift the chosen column downwards by one position
        acs = obs.iloc[:, max_cumsum_index].shift(1)
        # Drop the chosen column using its label
        obs = obs.drop(obs.columns[max_cumsum_index], axis=1)

        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    

    elif nme == 'mnist':

        # Define transformation to convert images to tensors and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts images to PyTorch tensors and normalizes to [0,1]
        ])

        # Load MNIST training dataset using torchvision
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

        # Specify the number of samples to use
        num_samples = 50

        # Option 1: Randomly select 7000 samples
        indices = torch.randperm(len(train_dataset))[:num_samples]
        subset_train_dataset = torch.utils.data.Subset(train_dataset, indices)

        # Option 2: Use the first 7000 samples (uncomment the following lines if you prefer this)
        # subset_train_dataset = torch.utils.data.Subset(train_dataset, range(num_samples))

        # Initialize lists to hold images and labels
        images = []
        labels = []

        # Extract images and labels from the subset dataset
        for img, label in subset_train_dataset:
            images.append(img.numpy())    # Convert tensor to NumPy array
            labels.append(label)

        # Convert lists to NumPy arrays
        x = np.array(images)   # Shape: (7000, 1, 28, 28)
        y = np.array(labels)   # Shape: (7000,)

        # Flatten the images (from 1 x 28 x 28 to 784)
        x = x.reshape((x.shape[0], -1))  # Shape: (7000, 784)

        # Ensure pixel values are float32
        x = x.astype(np.float32)

        # Save features and labels with consistent naming
        obs = x  # Features: shape (7000, 784)
        acs = y  # Labels: shape (7000,)

        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
    else:
        raise ValueError("Dataset name not recognized")


    return obs, acs


def process_neurons(df):
    # Convert column 0 to multiples of (16/1000) and take the floor to create bins
    df['bin'] = np.floor(df.iloc[:, 0] / (16/1000)).astype(int)

    # Initialize a list to store the results
    results = []

    # Group by the bin and process each group
    for _, group in df.groupby('bin'):
        occurred = [0] * 60
        for item in group.iloc[:, 1]:
            if 0 <= item < 60:
                occurred[int(item)] = 1
        results.append(occurred)

    # Create a new DataFrame from the results list
    results_df = pd.DataFrame(results, columns=[i for i in range(60)])

    return results_df


def generate_neuron_data(nme):
    # Regular expression to match the name pattern and extract x, y, and day
    match = re.match(r"timme(\d+)_to_(\d+)_neurons_day(\d+)", nme)
    if not match:
        print("The name does not match the required format.")
        return

    # Extract start_neuron, end_neuron, and day from the matched groups
    start_neuron = int(match.group(1))
    end_neuron = int(match.group(2))
    day = int(match.group(3))

    # Define the days of interest as a list
    days_of_interest = [4, 7, 12, 16, 20, 25, 31, 33]
    
    # Check if the specified day is in the days of interest
    if day not in days_of_interest:
        print(f"Day {day} is not in the days of interest.")
        return
    
    # Format the file name based on the day
    file_name = f"2-2-{day}.spk.txt"
    try:
        # Load the neuron spike data from the file
        obs = pd.read_csv(file_name, delimiter=" ")
        obs = process_neurons(obs).iloc[:, start_neuron-1:end_neuron]
        print(obs.shape)
        obs = obs.sample(frac=(1000/obs.shape[0]), replace=True, random_state=1)
        cumulative_sums = obs.cumsum().iloc[-1]
        print(obs.shape[0])
        # Find the column with the highest cumulative sum
        max_cumsum_col = start_neuron-1 + random.randint(0,9)
        print(max_cumsum_col)
        # Shift this column downwards by one position
        acs = obs[max_cumsum_col].shift(1)
        obs = obs.drop(max_cumsum_col, axis=1)


        obs = np.array(obs)  # shape: (batch_size, sequence_len, input_dim=3)
        acs = np.array(acs)
        np.save(f'obs_{nme}.npy', obs)
        np.save(f'acs_{nme}.npy', acs)
        
        return obs, acs
        
    except Exception as e:
        print(f"An error occurred: {e}")


