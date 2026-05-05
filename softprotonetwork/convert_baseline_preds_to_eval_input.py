NUM_TRIALS = 100

import torch
import numpy as np



for i in range(NUM_TRIALS):
    truth = np.load("../test_labels.npy")
    print(truth.shape)
    preds = torch.load(f"baseline_prototypical_model_preds_{i}.pt")
    print(preds.shape)
    combined_array = np.concatenate((truth, preds.numpy()), axis=1)
    print(combined_array.shape)
    print(preds)
    print(truth)
    np.savetxt(f"prototypical_model_eval_{i}.csv", combined_array, delimiter=",", fmt='%f')
    break