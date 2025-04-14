from synthetic_data import generate_synthetic
from data_augmentation import run_augmentation
from teed import main
from eval import run_inference, run_eval

generate_synthetic(10, 200)
run_augmentation('data/synthetic_train', rotation = True)
main()
run_inference()
run_eval()