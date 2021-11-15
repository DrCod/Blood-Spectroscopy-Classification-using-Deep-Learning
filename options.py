import argparse

def get_parser():
    
    parser = argparse.ArgumentParser(description = "Zindi Spectroscopy Classification Challenge")
    parser.add_argument("--train_csv", type = str,   default = 'Train.csv', help = "path/to/train csv")
    parser.add_argument("--test_csv", type = str,   default = 'Test.csv', help = "path/to/test csv")
    parser.add_argument("--num_env_features", type = int, default = 2, help = "numbe of scanner environment feature")
    parser.add_argument("--use_threshold", action='store_true', default=None, help='toggle between using thresholded features or full' )
    parser.add_argument("--use_smoothing", action='store_true', default=None, help='toggle to use smoothing' )
    parser.add_argument("--use_real_only", action='store_true', default=None, help='toggle to using only real component features' )
    parser.add_argument("--threshold", type = int,default = 10, help = "FFT signal dimension threshold")
    parser.add_argument("--BATCH_SIZE",type = int,   default = 32, help = "batch size")
    parser.add_argument("--EARLY_STOP", action='store_true', default=None, help='Early stop flag')
    parser.add_argument("--EPOCHS", type = int , default = 100, help = "number of epochs")
    parser.add_argument("--LR", type = float, default = 5e-3, help = "learning rate")
    parser.add_argument("--NFOLDS", type = int, default = 10, help ="Number of folds")
    parser.add_argument("--WEIGHT_DECAY", type = float, default = 1e-5, help = "weight decay")
    parser.add_argument("--EARLY_STOPPING_STEPS", type = int,  default = 10, help = "patience parameter")
    parser.add_argument("--hidden_size", type = int, default = 128, help = "hidden dimension size for FFT features")
    parser.add_argument("--hidden_size_env", type = int, default = 64, help = "hidden dimension size for scanner env features")
    parser.add_argument("--num_targets", type = int, default = 9, help = "number of targets")
    parser.add_argument("--smoothing", type =float , default = 0.001, help = "smoothing parameter")
    parser.add_argument("--model_output_folder", type =str, default = "data/", help = "model output directory")
    parser.add_argument("--model_name", type =str, default = "NNet", help = "model name")
    parser.add_argument("--model_type", type = str, default ="single", choices = ["single", "double"], help="Model architecture type" )
    
    return parser