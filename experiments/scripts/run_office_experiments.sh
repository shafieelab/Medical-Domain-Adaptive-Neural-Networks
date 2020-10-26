# Default configuration
seed=0
gpu_id=0
dset=office
arch=ResNet50
no_of_classes=31
output_dir="experiments"
crop_size=224
image_size=256



## run office experiments
#for trial in 1 2 3
#do
#  for source in Amazon Webcam DSLR
#  do
#    for target in Amazon Webcam DSLR
#    do
#      if [ ${source} == ${target} ]
#      then
#        continue
#      fi
#
#        printf "${source} to ${target} \n \n"
#        python -u experiments/python_scripts/train_md_nets.py --mode train \
#                      --seed $seed --num_iterations 20 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
#                      --dset ${dset} --s_dset ${source} --t_dset ${target} \
#                      --s_dset_txt "data/office/txt_files/${source}_source-${trial}.txt"\
#                      --sv_dset_txt "data/office/txt_files/${source}_validation-${trial}.txt" \
#                      --t_dset_txt "data/office/txt_files/${target}_target.txt" \
#                      --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch}\
#                       --crop_size ${crop_size} --image_size ${image_size}
#
#
#        printf "DONE \n \n \n "
#    done
#  done
#done
#



###################       Source Model to Target Domain Adaption       #############################################################
######################### MD-nets - using source and target data           ############################################
#######################################################################################################################

##########################      A --> W    #############################################################################

python -u experiments/python_scripts/train_md_nets.py --mode train \
        --dset office --s_dset A --t_dset W \
        --lr 0.001 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
        --num_iterations 100000 --patience 10000 --test_interval 500 --snapshot_interval 1000 \
        --s_dset_txt "data/office/txt_files/A90.txt"\
        --sv_dset_txt "data/office/txt_files/A10.txt" \
        --t_dset_txt  "data/office/txt_files/W.txt"  \
        --dset ${dset}  --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \
        --crop_size ${crop_size} --image_size ${image_size} --seed ${seed} \


##########################      D --> W    #############################################################################

python -u experiments/python_scripts/train_md_nets.py --mode train \
        --s_dset D --t_dset W \
        --lr 0.0003 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
        --num_iterations 100000 --patience 10000 --test_interval 500 --snapshot_interval 1000 \
        --s_dset_txt "data/office/txt_files/D90.txt"\
        --sv_dset_txt "data/office/txt_files/D10.txt" \
        --t_dset_txt  "data/office/txt_files/W.txt"  \
        --dset ${dset} --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \
         --crop_size ${crop_size} --image_size ${image_size} --seed ${seed} \


##########################      W --> D    #############################################################################

python -u experiments/python_scripts/train_md_nets.py --mode train \
        --dset office --s_dset W --t_dset D \
        --lr 0.001 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
        --num_iterations 100000 --patience 10000 --test_interval 500 --snapshot_interval 1000 \
        --s_dset_txt "data/office/txt_files/W90.txt"\
        --sv_dset_txt "data/office/txt_files/W10.txt" \
        --t_dset_txt  "data/office/txt_files/D.txt"  \
        --dset ${dset} --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \
         --crop_size ${crop_size} --image_size ${image_size} --seed ${seed} \


##########################      A --> D    #############################################################################

python -u experiments/python_scripts/train_md_nets.py --mode train \
        --dset office --s_dset A --t_dset D \
        --lr 0.0003 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
        --num_iterations 100000 --patience 10000 --test_interval 500 --snapshot_interval 1000 \
        --s_dset_txt "data/office/txt_files/A90.txt"\
        --sv_dset_txt "data/office/txt_files/A10.txt" \
        --t_dset_txt  "data/office/txt_files/D.txt"  \
        --dset ${dset} --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \
         --crop_size ${crop_size} --image_size ${image_size} --seed ${seed} \


##########################      D --> A    #############################################################################

python -u experiments/python_scripts/train_md_nets.py --mode train \
        --dset office --s_dset D --t_dset A \
        --lr 0.001 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
        --num_iterations 100000 --patience 10000 --test_interval 500 --snapshot_interval 1000 \
        --s_dset_txt "data/office/txt_files/D90.txt"\
        --sv_dset_txt "data/office/txt_files/D10.txt" \
        --t_dset_txt  "data/office/txt_files/A.txt"  \
        --dset ${dset}  --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \
         --crop_size ${crop_size} --image_size ${image_size} --seed ${seed} \

##########################      W --> A    #############################################################################

python -u experiments/python_scripts/train_md_nets.py --mode train \
        --dset office --s_dset W --t_dset A \
        --lr 0.001 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
        --num_iterations 100000 --patience 10000 --test_interval 500 --snapshot_interval 1000 \
        --s_dset_txt "data/office/txt_files/W90.txt"\
        --sv_dset_txt "data/office/txt_files/W10.txt" \
        --t_dset_txt  "data/office/txt_files/A.txt"  \
        --dset ${dset} --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \
         --crop_size ${crop_size} --image_size ${image_size} --seed ${seed} \


########################################################################################################################
##########################      ResNet50 Source only       #############################################################
########################################################################################################################

##########################      A Source Only     ######################################################################
python -u experiments/python_scripts/train_cnn.py --mode train \
              --seed $seed --num_iterations 10000 --patience 5000 --test_interval 50 --snapshot_interval 1000 \
              --s_dset A \
              --lr 0.02 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
              --s_dset_txt "data/office/txt_files/A90.txt" \
              --sv_dset_txt "data/office/txt_files/A10.txt"\
              --test_dset_txt "data/office/txt_files/A10.txt" \
              --dset ${dset}  --crop_size ${crop_size} --image_size ${image_size} --seed ${seed}\
              --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \

##########################      D Source Only     ######################################################################
python -u experiments/python_scripts/train_cnn.py --mode train \
              --seed $seed --num_iterations 10000 --patience 5000 --test_interval 50 --snapshot_interval 1000 \
              --dset office --s_dset D \
              --lr 0.01 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
              --s_dset_txt "data/office/txt_files/D90.txt" \
              --sv_dset_txt "data/office/txt_files/D10.txt"\
              --test_dset_txt "data/office/txt_files/D10.txt" \
               --crop_size ${crop_size} --image_size ${image_size} --seed ${seed}\
              --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \

##########################      W Source Only     ######################################################################
python -u experiments/python_scripts/train_cnn.py --mode train \
              --seed $seed --num_iterations 10000 --patience 5000 --test_interval 50 --snapshot_interval 1000 \
              --s_dset W \
              --lr 0.01 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
              --s_dset_txt "data/office/txt_files/W90.txt" \
              --sv_dset_txt "data/office/txt_files/W10.txt"\
              --test_dset_txt "data/office/txt_files/W10.txt" \
               --dset ${dset}  --crop_size ${crop_size} --image_size ${image_size} --seed ${seed}\
              --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \


########################################################################################################################
########################## MD-nets (NoS) Pre-trained Source Model to Target Domain Adaption ############################
################################################################ No source data used ###################################

##########################      A --> W    #############################################################################
python -u experiments/python_scripts/train_md_nets_nos.py --mode train \
        --s_dset A --t_dset W \
        --lr 0.0001 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
        --num_iterations 10000 --patience 2000 --test_interval 200 --snapshot_interval 1000 \
        --t_dset_txt  "data/office/txt_files/W.txt" --tv_dset_txt "data/office/txt_files/W10.txt" \
        --dset ${dset}  --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \
         --crop_size ${crop_size} --image_size ${image_size} --seed ${seed} \
       --trade_off_cls 0.7 --source_model_path "experiments/models/office/ResNet50/train_A_CNN/best_model.pth.tar"

##########################      D --> W    #############################################################################
python -u experiments/python_scripts/train_md_nets_nos.py --mode train \
        --s_dset D --t_dset W \
        --lr 0.001 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
        --num_iterations 10000 --patience 10000 --test_interval 200 --snapshot_interval 1000 \
        --t_dset_txt  "data/office/txt_files/W.txt" --tv_dset_txt "data/office/txt_files/W10.txt" \
        --dset ${dset} --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \
         --crop_size ${crop_size} --image_size ${image_size} --seed ${seed} \
       --trade_off_cls 0.3 --source_model_path "experiments/models/office/ResNet50/train_D_CNN/best_model.pth.tar"


##########################      W --> D    #############################################################################
python -u experiments/python_scripts/train_md_nets_nos.py --mode train \
        --s_dset W --t_dset D \
        --lr 0.001 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
        --num_iterations 10000 --patience 10000 --test_interval 200 --snapshot_interval 1000 \
        --t_dset_txt  "data/office/txt_files/D.txt" --tv_dset_txt "data/office/txt_files/D10.txt" \
        --dset ${dset} --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \
         --crop_size ${crop_size} --image_size ${image_size} --seed ${seed} \
        --trade_off_cls 0.3  --source_model_path "experiments/models/office/ResNet50/train_W_CNN/best_model.pth.tar"


##########################      A --> D    #############################################################################
python -u experiments/python_scripts/train_md_nets_nos.py --mode train \
        --s_dset A --t_dset D \
        --lr 0.001 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
        --num_iterations 10000 --patience 10000 --test_interval 200 --snapshot_interval 1000 \
        --t_dset_txt  "data/office/txt_files/D.txt" --tv_dset_txt "data/office/txt_files/D10.txt" \
        --dset ${dset} --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \
         --crop_size ${crop_size} --image_size ${image_size} --seed ${seed} \
        --trade_off_cls 1.0  --source_model_path "experiments/models/office/ResNet50/train_A_CNN/best_model.pth.tar"


##########################      D --> A    #############################################################################
python -u experiments/python_scripts/train_md_nets_nos.py --mode train \
        --s_dset D --t_dset A \
        --lr 0.0001 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
        --num_iterations 10000 --patience 10000 --test_interval 200 --snapshot_interval 1000 \
        --t_dset_txt  "data/office/txt_files/A.txt" --tv_dset_txt "data/office/txt_files/A10.txt" \
       --dset ${dset}  --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \
         --crop_size ${crop_size} --image_size ${image_size} --seed ${seed} \
        --trade_off_cls 1.0  --source_model_path "experiments/models/office/ResNet50/train_D_CNN/best_model.pth.tar"

##########################      W --> A    #############################################################################
python -u experiments/python_scripts/train_md_nets_nos.py --mode train \
        --s_dset W --t_dset A \
        --lr 0.0001 --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256 \
        --num_iterations 10000 --patience 10000 --test_interval 200 --snapshot_interval 1000 \
        --t_dset_txt  "data/office/txt_files/A.txt" --tv_dset_txt "data/office/txt_files/A10.txt" \
        --dset ${dset} --no_of_classes ${no_of_classes} --output_dir ${output_dir}  --gpu_id ${gpu_id} --arch ${arch} \
         --crop_size ${crop_size} --image_size ${image_size} --seed ${seed} \
        --trade_off_cls 1.0   --source_model_path "experiments/models/office/ResNet50/train_W_CNN/best_model.pth.tar"






