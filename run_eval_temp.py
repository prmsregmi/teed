
import subprocess
def call_ods_ois(model, mat_dir):
    # Path to the external Python interpreter if different from the main project's
    external_python = ".venv/bin/python3"
    
    # Build the command
    cmd = [
        external_python,
        "main.py",
        "--alg", "TEED " + model,
        "--model_name_list", model,
        "--result_dir", "../../" + mat_dir,
        "--save_dir", "../../" + mat_dir + "/eval_output",
        "--gt_dir", "../../data/UDED/gt_mat",
        "--key", "result",
        "--file_format", ".mat",
        "--workers", "-1"
    ]
    
    # Call the script, changing to its directory
    subprocess.run(cmd, cwd="external/edge_eval")


for i in range(7):
    mat_dir = f"checkpoints/CLASSIC/{i}/UDED_res/result_mat"
    call_ods_ois("Classic", mat_dir)