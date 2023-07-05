# Before running these commands, please read and run "optimizees/lasso_bsds500.py"

# [Generating ground truth]
python main.py --config ./configs/3_lasso_testing_real.yaml --optimizee-dir ./optimizees/matdata/lasso-real --load-mat --save-sol --optimizer ProximalGradientDescentMomentum --save-dir LASSO-FISTA --test-length 5000

# [Testing L2O models]
python main.py --config ./configs/3_lasso_testing_real.yaml --optimizee-dir ./optimizees/matdata/lasso-real --load-mat --load-sol --p-use --a-use --save-dir LASSO-L2O-PA
python main.py --config ./configs/3_lasso_testing_real.yaml --optimizee-dir ./optimizees/matdata/lasso-real --load-mat --load-sol --optimizer CoordBlackboxLSTM --grad-method bp_grad --save-dir LASSO-L2O-DM
python main.py --config ./configs/3_lasso_testing_real.yaml --optimizee-dir ./optimizees/matdata/lasso-real --load-mat --load-sol --optimizer RNNprop --grad-method bp_grad --save-dir LASSO-L2O-RNNprop

# [Testing hand-designed optimizers]
python main.py --config ./configs/3_lasso_testing_real.yaml --optimizee-dir ./optimizees/matdata/lasso-real --load-mat --load-sol --optimizer ProximalGradientDescentMomentum --save-dir LASSO-FISTA
python main.py --config ./configs/3_lasso_testing_real.yaml --optimizee-dir ./optimizees/matdata/lasso-real --load-mat --load-sol --optimizer ProximalGradientDescent --save-dir LASSO-ISTA
python main.py --config ./configs/3_lasso_testing_real.yaml --optimizee-dir ./optimizees/matdata/lasso-real --load-mat --load-sol --optimizer Adam --step-size 1e-3 --momentum1 1e-4 --momentum2 1e-1 --save-dir LASSO-Adam
python main.py --config ./configs/3_lasso_testing_real.yaml --optimizee-dir ./optimizees/matdata/lasso-real --load-mat --load-sol --optimizer AdamHD --step-size 1e-2 --momentum1 1e-2 --momentum2 1e-2 --hyper-step 1e-08 --save-dir LASSO-AdamHD
