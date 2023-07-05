
# [generating test instances and the solutions and saving them]

## generate L1-regularized Logistic Regression instances and save them to "./optimizees/matdata/logistic-rand"
python main.py --config ./configs/logistic_testing.yaml --optimizer ProximalGradientDescentMomentum --save-dir LogisticL1-FISTA --save-to-mat --optimizee-dir ./optimizees/matdata/logistic-rand

## solve L1-regularized Logistic Regression with FISTA and save the optimal objective value for each instance (5,000 iterations are sufficient to obtain optimal objective)
python main.py --config ./configs/logistic_testing.yaml --optimizer ProximalGradientDescentMomentum --save-dir LogisticL1-FISTA --load-mat --save-sol --optimizee-dir ./optimizees/matdata/logistic-rand --test-length 5000


# [train models for out method, L2O-DM and L2O-RNNprop]
python main.py --config ./configs/logistic_training.yaml --p-use --a-use --save-dir LogisticL1-PA
python main.py --config ./configs/logistic_training.yaml --optimizer CoordBlackboxLSTM --grad-method bp_grad --save-dir LogisticL1-L2O-DM
python main.py --config ./configs/logistic_training.yaml --optimizer RNNprop --grad-method bp_grad --save-dir LogisticL1-L2O-RNNprop


# [test L2O-DM and L2O-RNNprop]
python main.py --config ./configs/logistic_testing.yaml --p-use --a-use --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-rand --save-dir LogisticL1-PA
python main.py --config ./configs/logistic_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-rand --optimizer CoordBlackboxLSTM --grad-method bp_grad --save-dir LogisticL1-L2O-DM
python main.py --config ./configs/logistic_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-rand --optimizer RNNprop --grad-method bp_grad --save-dir LogisticL1-L2O-RNNprop


# [train and test models for Ada-LISTA] //This may take long time for problems with size of 250*500.
python main_unroll.py --optimizer AdaLISTA --optimizee-type LogisticL1 --input-dim 50 --sparsity 20 --output-dim 1000 --layers 10 --init-lr 2e-3 --save-dir LogisticL1-AdaLISTA
python main_unroll.py --config ./configs/logistic_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-rand --optimizer AdaLISTA --layers 10 --init-lr 2e-3 --save-dir LogisticL1-AdaLISTA


# [test other hand-designed optimizers]
python main.py --config ./configs/logistic_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-rand --optimizer ProximalGradientDescentMomentum --save-dir LogisticL1-FISTA
python main.py --config ./configs/logistic_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-rand --optimizer ProximalGradientDescent --save-dir LogisticL1-ISTA
python main.py --config ./configs/logistic_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-rand --optimizer Adam --step-size 1e-2 --momentum1 1e-1 --momentum2 1e-1 --save-dir LogisticL1-Adam
python main.py --config ./configs/logistic_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-rand --optimizer AdamHD --step-size 0.1 --momentum1 0.001 --momentum2 0.1 --hyper-step 1e-07 --save-dir LogisticL1-AdamHD
