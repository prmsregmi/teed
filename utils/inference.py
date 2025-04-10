import os
import toml
from teed.main import main


def run_inference():
    # Get all folders starting with "checkpoints" in current directory
    checkpoint_dirs = [d for d in os.listdir(".") if d.startswith("checkpoints") and os.path.isdir(d)]
    original_config = toml.load("config.toml")
    # Loop through each checkpoints folder
    for checkpoint_dir in checkpoint_dirs:        
        # Loop through epoch folders 0,2,4,6
        for i in range(0, 7):
            config = toml.load("config.toml")
            config["general"]["is_testing"] = True
            config["training"]["checkpoint_data"] = f"{i}_model.pth"
            config["paths"]["output_dir"] = f"{checkpoint_dir}/CLASSIC/{i}/"
            with open("config.toml", "w") as f:
                toml.dump(config, f)
            main()
    
    # Restore original config
    with open("config.toml", "w") as f:
        toml.dump(original_config, f)