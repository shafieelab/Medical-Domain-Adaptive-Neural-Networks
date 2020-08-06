seed=0
# run office experiments
for trial in 1 2 3
do
  for source in Amazon Webcam DVD
  do
    for target in Amazon Webcam DVD
    do
      if [ ${source} == ${target} ]
      then
        continue
      fi

        printf "${source} to ${target} \n \n"
        python -u experiments/tools/train_md_nets.py --mode train \
                      --seed $seed --num_iterations 10 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                      --dset office --s_dset ${source} --t_dset ${target} \
                      --s_dset_txt "data/office/txt_files/${source}_source-${trial}.txt"\
                      --sv_dset_txt "data/office/txt_files/${source}_validation-${trial}.txt" \
                      --t_dset_txt "data/office/txt_files/${target}_target.txt" \
                      --no_of_classes 31 --output_dir "experiments"  --gpu_id "0" --arch Xception\
                      --target_labelled true --crop_size 256 --image_size 224


        printf "DONE \n \n \n "
    done
  done
done

