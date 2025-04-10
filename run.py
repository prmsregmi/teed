from synthetic_data import generate_synthetic
from data_augmentation import run_augmentation
from teed import main

generate_synthetic(35, 165)
run_augmentation('data/synthetic_train', rotation = True)
main()