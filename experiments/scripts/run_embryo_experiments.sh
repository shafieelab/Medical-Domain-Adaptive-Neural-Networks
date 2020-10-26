########################################################################################################################
########################## MD-nets Source to Target Domain Adaption ####################################################
########################################################################################################################



seed=0
# run embryo same domain settings
# ED4 to ED4 adaption

python -u experiments/python_scripts/train_md_nets.py --mode train \
                  --seed $seed --num_iterations 50000 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                  --dset embryo  --s_dset ed4 --t_dset ed4 --lr 0.01\
                  --s_dset_txt "data/embryo/ed4/ed4_source_same_domain.txt" \
                  --sv_dset_txt "data/embryo/ed4/ed4_validation.txt" \
                  --t_dset_txt "data/embryo/ed4/ed4_target_same_domain.txt" \
                  --no_of_classes 5 --output_dir "experiments" --gpu_id 0 --arch Xception\
                            --crop_size 224 --image_size 256
# Embryo cross domain settings
# ED4 to ED3 adaption
python -u experiments/python_scripts/train_md_nets.py --mode train \
                  --seed $seed --num_iterations 50000 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                  --dset embryo  --s_dset ed4 --t_dset ed3  --lr 0.01\
                  --s_dset_txt "data/embryo/ed4/ed4_source_2-class.txt" \
                  --sv_dset_txt "data/embryo/ed4/ed4_validation_2-class.txt" \
                  --t_dset_txt "data/embryo/ed3/ed3_target.txt" \
                  --no_of_classes 2 --output_dir "experiments"  --gpu_id 0 --arch Xception\
                            --crop_size 224 --image_size 256

# Embryo cross domain settings
# ED4 to ED2 and ED1 adaption
for target in ed1 ed2
do
    python -u experiments/python_scripts/train_md_nets.py --mode train \
                  --seed $seed --num_iterations 50000 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                  --dset embryo --s_dset ed4 --t_dset $target  --lr 0.01 \
                  --s_dset_txt "data/embryo/ed4/ed4_source.txt" --sv_dset_txt "data/embryo/ed4/ed4_validation.txt" \
                  --t_dset_txt "data/embryo/${target}/${target}_target.txt" \
                  --no_of_classes 5 --output_dir "experiments"  --gpu_id 0 --arch Xception\
                  --crop_size 224 --image_size 256

done