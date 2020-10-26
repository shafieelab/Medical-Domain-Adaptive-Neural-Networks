seed=0
# run embryo cross domain settings
# ED4 to ED2 and ED1 adaption
for target in ed1 ed2
do
    python -u -W ignore  experiments/python_scripts/train_md_nets.py \
          --mode test \
          --seed $seed --dset embryo --s_dset ed4 --t_dset $target \
          --s_dset_txt "data/embryo/ed4/ed4_source.txt" --sv_dset_txt "data/embryo/ed4/ed4_validation.txt" \
          --t_dset_txt "data/embryo/${target}/${target}_target.txt" \
          --no_of_classes 5 --output_dir "experiments"  --gpu_id 1 --arch Xception\
          --crop_size 224 --image_size 256 \
          --trained_model_path "experiments/submitted_models/embryo/Xception/train_ed4_to_${target}/best_model.pth.tar"

    printf "DONE \n \n \n "
    printf "\n \n \n "

done
# ED4 to ED3 adaption
python -u -W ignore  experiments/python_scripts/train_md_nets.py \
                  --mode test \
                  --seed $seed --dset embryo  --s_dset ed4 --t_dset ed3 \
                  --s_dset_txt "data/embryo/ed4/ed4_source_2-class.txt" \
                  --sv_dset_txt "data/embryo/ed4/ed4_validation_2-class.txt" \
                  --t_dset_txt "data/embryo/ed3/ed3_target.txt" \
                  --no_of_classes 2 --output_dir "experiments"  --gpu_id 1 --arch Xception\
                  --crop_size 224 --image_size 256 \
          --trained_model_path "experiments/submitted_models/embryo/Xception/train_ed4_to_ed3/best_model.pth.tar"

printf "DONE \n \n \n "
printf " \n \n \n "


# run embryo same domain settings
# ED4 to ED4 adaption

python -u -W ignore  experiments/python_scripts/train_md_nets.py --mode test \
                  --seed $seed --dset embryo  --s_dset ed4 --t_dset ed4 \
                  --s_dset_txt "data/embryo/ed4/ed4_source_same_domain.txt" \
                  --sv_dset_txt "data/embryo/ed4/ed4_validation.txt" \
                  --t_dset_txt "data/embryo/ed4/ed4_target_same_domain.txt" \
                  --no_of_classes 5 --output_dir "experiments" --gpu_id 1 --arch Xception\
                  --crop_size 224 --image_size 256 \
          --trained_model_path "experiments/submitted_models/embryo/Xception/train_ed4_to_ed4/best_model.pth.tar"


printf "DONE \n \n \n "
printf "\n \n \n "
