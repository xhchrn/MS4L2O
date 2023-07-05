
# [train models for L2O-DM and L2O-RNNprop]
python main.py --config ./configs/1_lasso_training.yaml --optimizer CoordBlackboxLSTM --grad-method bp_grad --save-dir LASSO-L2O-DM
python main.py --config ./configs/1_lasso_training.yaml --optimizer RNNprop --init-lr 3e-3 --val-freq 5 --grad-method bp_grad --save-dir LASSO-L2O-RNNprop

# [test L2O-DM and L2O-RNNprop]
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand --optimizer CoordBlackboxLSTM --grad-method bp_grad --save-dir LASSO-L2O-DM
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand --optimizer RNNprop --grad-method bp_grad --save-dir LASSO-L2O-RNNprop

# [train and test models for Ada-LISTA] //This may take long time for problems with size of 250*500.
python main_unroll.py --optimizer AdaLISTA --optimizee-type LASSO --input-dim 500 --sparsity 50 --output-dim 250 --layers 10 --init-lr 2e-3 --save-dir LASSO-AdaLISTA
python main_unroll.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand --optimizer AdaLISTA --layers 10 --init-lr 2e-3 --save-dir LASSO-AdaLISTA

# [test other hand-designed optimizers]
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand --optimizer ProximalGradientDescentMomentum --save-dir LASSO-FISTA
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand --optimizer ProximalGradientDescent --save-dir LASSO-ISTA
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand --optimizer Adam --step-size 1e-2 --momentum1 1e-1 --momentum2 1e-1 --save-dir LASSO-Adam
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand --optimizer AdamHD --step-size 0.1 --momentum1 0.001 --momentum2 0.1 --hyper-step 1e-07 --save-dir LASSO-AdamHD