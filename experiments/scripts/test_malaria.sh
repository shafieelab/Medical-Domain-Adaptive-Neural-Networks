seed=0


# run Malaria CNN settings
# MD4 to MD4 adaption
printf "MD4 CNN model \n \n"

python -u experiments/tools/train_cnn.py --mode test \
                  --seed $seed --num_iterations 10 --patience 5000 --test_interval 50 --snapshot_interval 1000 \
                  --dset malaria  --s_dset MD4 \
                  --s_dset_txt "data/malaria/MD4/txt_70_20_10/MD4_random_train.txt" \
                  --sv_dset_txt "data/malaria/MD4/txt_70_20_10/MD4_random_valid.txt"\
                  --test_dset_txt "data/malaria/MD4/txt_70_20_10/MD4_random_test.txt" \
                  --crop_size 100 --image_size 100 \
                  --no_of_classes 2 --output_dir "experiments" --gpu_id "0" --arch ResNet50\
                  --trained_model_path "experiments/submitted_models/malaria/ResNet50/train_MD4_CNN/best_model.pth.tar"

printf "DONE \n \n \n "



# run Malaria pre-trained model settings
# MD4 to MD3, MD2 and MD1 pre-trained model adaption
for target in MD3 MD2 MD1
do
    printf "MD4 to ${target} \n \n"

    python -u experiments/tools/train_md_nets.py --mode test \
                  --seed $seed --num_iterations 10 --patience 5000 --test_interval 50 --snapshot_interval 1000 \
                  --dset malaria --s_dset md4 --t_dset $target \
                  --s_dset_txt "data/malaria/MD4/txt_70_20_10/MD4_random_valid.txt"\
                  --sv_dset_txt "data/malaria/MD4/txt_70_20_10/MD4_random_test.txt" \
                  --t_dset_txt "data/malaria/${target}_target.txt" \
                  --test_dset_txt "data/malaria/${target}_test.txt" \
                  --no_of_classes 2 --output_dir "experiments"  --gpu_id "0" --arch ResNet50\
                  --target_labelled true --crop_size 100 --image_size 100 \
                  --trained_model_path "experiments/submitted_models/malaria/ResNet50/train_MD4_to_${target}/best_model.pth.tar"
                  # trained model from CNN settings

    printf "DONE \n \n \n "

done



