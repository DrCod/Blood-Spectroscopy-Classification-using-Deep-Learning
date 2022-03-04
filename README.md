# Blood-Spectroscopy-Classification-using-Deep-Learning

A starter solution code(including the best final solution iteration) to a machine learning problem from Zindi.Africa.

Final notebook - https://github.com/DrCod/Blood-Spectroscopy-Classification-using-Deep-Learning/blob/main/Final%20Solution%20Implementation.ipynb

Starter iteration utilized FFT features to train a neural network architecture in a single/multi-input fashion.

The ultimate goal of this project was to accurately identify the levels of different blood-based substrates from Near-Infrared Reflectance(NIR) spectral data.

More info about the challenge and datasets acquisition can be found here - https://zindi.africa/competitions/bloodsai-blood-spectroscopy-classification-challenge

# How to ran starter scripts in single/multi-input modes 

- Using single input:

  python train.py --train_csv Train.csv --test_csv Test.csv --use_threshold --use_smoothing --BATCH_SIZE  32 \
                  --EARLY_STOP --EPOCHS 100 --WEIGHT_DECAY 1e-6 --model_type single --model_name NN
                
        
- Using multi-input:

  python train.py --train_csv Train.csv --test_csv Test.csv --use_threshold --use_smoothing --BATCH_SIZE  32 \
                  --EARLY_STOP --EPOCHS 100 --WEIGHT_DECAY 1e-6 --model_type double --model_name NN

Run submission inference:

python make_submission.py --model_name NN --submission_dir path/to/submissions
