from synthetic_data import generate_synthetic
from data_augmentation import run_augmentation
from teed import main
from eval import run_eval_batch
import shutil
import os
for i in range(3):
    if os.path.exists('data/synthetic_train'):
        shutil.rmtree('data/synthetic_train')
    generate_synthetic(200, 3000)
    run_augmentation('data/synthetic_train', rotation = False)
    main(i)
    # run_eval_batch(folder_name = f'checkpoints/checkpoints_{i}', multithread=False)