Example:
MPFL
```
python MPFL_2_2.py --comm_round_num 100 --client_num 600 --activate_proportion 0.03 --agg_proportion 0.5 --test_interval 10 --multi_process 0 --model lstm_shakespeare --dataset shakespeare --learning_rate 0.8 --batch_size 100 --epoch_num 1 --population_num 3 --immigration_interval 1 --immigration_num 1 --competition_threshold 0.8 --election_interval 1 --malicious_proportion 0 --malicious_attack no_attack --noise_intensity 0 --iid 0 --save_log 1 --random_seed 1 --log_name mpfl-test
```

FL
```
python FL.py --comm_round_num 100 --client_num 200 --activate_proportion 0.1 --agg_proportion 0.5 --test_interval 10 --multi_process 0 --model lstm_shakespeare --dataset shakespeare --learning_rate 0.5 --batch_size 100 --epoch_num 1 --malicious_proportion 0 --malicious_attack no_attack --defense_algorithm fed_avg --noise_intensity 0 --iid 0 --save_log 1 --random_seed 1 --log_name fl-test
```
