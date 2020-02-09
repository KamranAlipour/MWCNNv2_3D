#python main.py --model MWCNN --save MWCNN_DeNoising --scale 15 --n_feats 64 --save_results --print_model --n_colors 1 --test_only --self_ensemble --resume -1 --pre_train experiment/MWCNN_DeNoising_backup/model/ --task_type denoising --generate True --gen_set "../data/ucsdLED_1k_npy/*"
#mv results_\* ucsdLED_1k_results
python main.py --model MWCNN --save MWCNN_DeNoising --scale 15 --n_feats 64 --save_results --print_model --n_colors 1 --test_only --self_ensemble --resume -1 --pre_train experiment/MWCNN_DeNoising_backup/model/ --task_type denoising --generate True --gen_set "../data/ucsdLED_2k_npy/*"
mv experiment/MWCNN_DeNoising/results_\* experiment/MWCNN_DeNoising/ucsdLED_2k_results
python main.py --model MWCNN --save MWCNN_DeNoising --scale 15 --n_feats 64 --save_results --print_model --n_colors 1 --test_only --self_ensemble --resume -1 --pre_train experiment/MWCNN_DeNoising_backup/model/ --task_type denoising --generate True --gen_set "../data/ucsdLED_4k_npy/*"
mv experiment/MWCNN_DeNoising/results_\* experiment/MWCNN_DeNoising/ucsdLED_4k_results
