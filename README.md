# Blood-Spectroscopy-Classification-using-Deep-Learning
My solution code to a machine learning problem from Zinid.Africa.
My approach utilizes FFT features to train a neural network architecture in a single/multi-input fashion.

Although I am continuously thinking of different angles to arrive at an optimal solution, I believe that FFT features and unsupervised feature learning techniques will be worth exploring further ,whilst bearing in mind the issue of overfitting.

The ultimate goal of this project is to accurately identify the status(high/low/okay) of the substrates of interest in every given blood spectrum.

More info about the challenge - https://zindi.africa/competitions/bloodsai-blood-spectroscopy-classification-challenge

Train examples:

- Using single input:

  python train.py --train_csv Train.csv --test_csv Test.csv --use_threshold --use_smoothing --BATCH_SIZE  32 \
                  --EARLY_STOP --EPOCHS 100 --WEIGHT_DECAY 1e-6 --model_type single --model_name NN
                
        
- Using multi-input:

  python train.py --train_csv Train.csv --test_csv Test.csv --use_threshold --use_smoothing --BATCH_SIZE  32 \
                  --EARLY_STOP --EPOCHS 100 --WEIGHT_DECAY 1e-6 --model_type double --model_name NN

Run submission inference:

python make_submission.py --model_name NN --submission_dir path/to/submissions
