

# [training commands] 

python main.py --config ./configs/1_lasso_training.yaml --a-use --save-dir LASSO-L2O-A 
python main.py --config ./configs/1_lasso_training.yaml --p-use --save-dir LASSO-L2O-P 
python main.py --config ./configs/1_lasso_training.yaml --p-use --a-use --save-dir LASSO-L2O-PA 
python main.py --config ./configs/1_lasso_training.yaml --p-use --b-use --a-use --save-dir LASSO-L2O-PBA 
python main.py --config ./configs/1_lasso_training.yaml --p-use --b-use --a-use --b1-use --save-dir LASSO-L2O-PBA1 
python main.py --config ./configs/1_lasso_training.yaml --p-use --b-use --a-use --b2-use --save-dir LASSO-L2O-PBA2 
python main.py --config ./configs/1_lasso_training.yaml --p-use --b-use --a-use --b1-use --b2-use --save-dir LASSO-L2O-PBA12 

# [generating test instances and save them]

## generate LASSO instances and save them to "./optimizees/matdata/lasso-rand"
python main.py --config ./configs/2_lasso_testing.yaml --optimizer ProximalGradientDescentMomentum --save-dir LASSO-FISTA --save-to-mat --optimizee-dir ./optimizees/matdata/lasso-rand

## solve LASSO with FISTA and save the optimal objective value for each instance (5000 iterations are sufficient to obtain optimal objective)
python main.py --config ./configs/2_lasso_testing.yaml --optimizer ProximalGradientDescentMomentum --save-dir LASSO-FISTA --load-mat --save-sol --optimizee-dir ./optimizees/matdata/lasso-rand --test-length 5000

# [test commands]

python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --a-use --save-dir LASSO-L2O-A --optimizee-dir ./optimizees/matdata/lasso-rand
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --p-use --save-dir LASSO-L2O-P --optimizee-dir ./optimizees/matdata/lasso-rand
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --p-use --a-use --save-dir LASSO-L2O-PA --optimizee-dir ./optimizees/matdata/lasso-rand
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --p-use --b-use --a-use --save-dir LASSO-L2O-PBA --optimizee-dir ./optimizees/matdata/lasso-rand
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --p-use --b-use --a-use --b1-use --save-dir LASSO-L2O-PBA1 --optimizee-dir ./optimizees/matdata/lasso-rand
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --p-use --b-use --a-use --b2-use --save-dir LASSO-L2O-PBA2 --optimizee-dir ./optimizees/matdata/lasso-rand
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --p-use --b-use --a-use --b1-use --b2-use --save-dir LASSO-L2O-PBA12 --optimizee-dir ./optimizees/matdata/lasso-rand
