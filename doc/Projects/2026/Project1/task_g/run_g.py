from config import CONFIG, RUN_FIT, RUN_JAX, RUN_KFOLD
from fitting import main_fit 
from kfold import kfold_function
from jax_ad import jax_function

if __name__ == "__main__":
    CONFIG["save_img"] = True
    CONFIG["save_csv"] = False

    if RUN_FIT:
        main_fit(save_figure=CONFIG.get("save_img", False))
    
    #run cv
    if RUN_KFOLD:
        kfold_function()
    
    #JAX experiment ONLY if enabled in config
    if RUN_JAX:
        jax_function()