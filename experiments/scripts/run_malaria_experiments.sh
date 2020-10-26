
seed=0
gpu_id=0
dset=malaria
arch=ResNet50
no_of_classes=2
output_dir="experiments"
crop_size=100
image_size=100

#########################################################################################################################
###########################      ResNet50 Source only       #############################################################
#########################################################################################################################
#
# run Malaria CNN settings
# MD4 to MD4 adaption
printf "MD4 CNN model \n \n"

python -u experiments/python_scripts/train_cnn.py --mode train \
                  --seed $seed --num_iterations 10000 --patience 5000 --test_interval 50 --snapshot_interval 1000 \
                  --dset malaria  --s_dset MD4 --use_bottleneck true \
                  --s_dset_txt "data/malaria/MD4/txt_70_20_10/MD4_train.txt" \
                  --sv_dset_txt "data/malaria/MD4/txt_70_20_10/MD4_valid.txt"\
                  --test_dset_txt "data/malaria/MD4/txt_70_20_10/MD4_test.txt" \
                   --dset ${dset}  --crop_size ${crop_size} --image_size ${image_size} --seed ${seed}\
                  --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \

printf "DONE \n \n \n "




########################################################################################################################
########################## MD-nets (NoS) Pre-trained Source Model to Target Domain Adaption ############################
################################################################ No source data used ###################################



python -u experiments/python_scripts/train_md_nets_nos.py --mode train \
        --s_dset MD4 --t_dset MD3 \
        --lr 0.001 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
        --num_iterations 10000 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
        --t_dset_txt  "data/malaria/MD3/MD3_target.txt" --tv_dset_txt "data/malaria/MD3/MD3_10_percent_validation.txt" \
        --dset ${dset}  --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \
         --crop_size ${crop_size} --image_size ${image_size} --seed ${seed} \
       --trade_off_cls 0.6 --source_model_path "experiments/models/malaria/ResNet50/train_MD4_CNN/best_model.pth.tar"



python -u experiments/python_scripts/train_md_nets_nos.py --mode train \
        --s_dset MD4 --t_dset MD2 \
        --lr 0.001 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
        --num_iterations 10000 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
        --t_dset_txt  "data/malaria/MD2/MD2_target.txt" --tv_dset_txt "data/malaria/MD2/MD2_10_percent_validation.txt" \
        --dset ${dset}  --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \
         --crop_size ${crop_size} --image_size ${image_size} --seed ${seed} \
       --trade_off_cls 1.0 --source_model_path "experiments/models/malaria/ResNet50/train_MD4_CNN/best_model.pth.tar"



python -u experiments/python_scripts/train_md_nets_nos.py --mode train \
        --s_dset MD4 --t_dset MD1 \
        --lr 0.0005 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
        --num_iterations 10000 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
        --t_dset_txt  "data/malaria/MD1/MD1_target.txt" --tv_dset_txt "data/malaria/MD1/MD1_10_percent_validation.txt" \
        --dset ${dset}  --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \
         --crop_size ${crop_size} --image_size ${image_size} --seed ${seed} \
       --trade_off_cls  0.05 --source_model_path "experiments/models/malaria/ResNet50/train_MD4_CNN/best_model.pth.tar"







## run Malaria pre-trained model settings
## MD4 to MD3, MD2 and MD1 pre-trained model adaption with source data
#for target in MD3 MD2 MD1
#do
#    printf "MD4 to ${target} \n \n"
#
#    python -u experiments/tools/train_md_nets.py --mode train \
#                  --seed $seed --num_iterations 10 --patience 5000 --test_interval 50 --snapshot_interval 1000 \
#                  --dset malaria --s_dset md4 --t_dset $target \
#                  --s_dset_txt "data/malaria/MD4/txt_70_20_10/MD4_random_valid.txt"\
#                  --sv_dset_txt "data/malaria/MD4/txt_70_20_10/MD4_random_test.txt" \
#                  --t_dset_txt "data/malaria/${target}_target.txt" \
#                  --test_dset_txt "data/malaria/${target}_test.txt" \
#                  --no_of_classes 2 --output_dir "experiments"  --gpu_id "0" --arch ResNet50\
#                  --target_labelled true --crop_size 100 --image_size 100 \
#                  --trained_model_path "experiments/models/malaria/ResNet50/train_MD4_CNN/best_model.pth.tar"
#                  # trained model from CNN settings
#
#    printf "DONE \n \n \n "
#
#done
#
#




