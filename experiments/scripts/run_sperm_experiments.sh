seed=0
# run sperm unlabelled settings
# SD4 to SD3, SD2 and SD4 adaption
printf "sd4 to sd3 \n \n"

python -u experiments/tools/train_md_nets.py --mode train \
              --seed $seed --num_iterations 10 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
              --dset sperm --s_dset sd4 --t_dset sd3_slides \
              --s_dset_txt "data/sperm/sd4/sd4_source.txt" --sv_dset_txt "data/sperm/sd4/sd4_validation.txt" \
              --t_dset_txt "data/sperm/sd3/sd3_patient_target_temp.txt" \
              --no_of_classes 2 --output_dir "experiments"  --gpu_id "0" --arch Xception\
              --target_labelled false \
              --sperm_patient_data_clinicians_annotations 'data/sperm/sperm_patient_data_clinicians_annotations.csv'

printf "DONE \n \n \n "


printf "sd4 to sd2 \n \n"

python -u experiments/tools/train_md_nets.py --mode train \
              --seed $seed --num_iterations 10 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
              --dset sperm --s_dset sd4 --t_dset sd2_slides \
              --s_dset_txt "data/sperm/sd4/sd4_source.txt" --sv_dset_txt "data/sperm/sd4/sd4_validation.txt" \
              --t_dset_txt "data/sperm/sd2/sd2_patient_target_temp.txt" \
              --no_of_classes 2 --output_dir "experiments"  --gpu_id "0" --arch Xception\
              --target_labelled false \
              --sperm_patient_data_clinicians_annotations 'data/sperm/sperm_patient_data_clinicians_annotations.csv'

printf "DONE \n \n \n "


printf "sd4 to sd1 \n \n"

python -u experiments/tools/train_md_nets.py --mode train \
              --seed $seed --num_iterations 10 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
              --dset sperm --s_dset sd4 --t_dset sd1_slides \
              --s_dset_txt "data/sperm/sd4/sd4_source.txt" --sv_dset_txt "data/sperm/sd4/sd4_validation.txt" \
              --t_dset_txt "data/sperm/sd1/sd1_patient_target_temp.txt" \
              --no_of_classes 2 --output_dir "experiments"  --gpu_id "0" --arch Xception\
              --target_labelled false \
              --sperm_patient_data_clinicians_annotations 'data/sperm/sperm_patient_data_clinicians_annotations.csv'

printf "DONE \n \n \n "







# run sperm labelled settings
# SD4 to SD4 adaption
printf "SD4 to ${target} Labelled \n \n"

python -u experiments/tools/train_md_nets.py --mode train \
                  --seed $seed --num_iterations 10 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                  --dset sperm  --s_dset sd4 --t_dset sd4 \
                  --s_dset_txt "data/sperm/sd4/same_domain/sd4_source_same_domain.txt" \
                  --sv_dset_txt "data/sperm/sd4/same_domain/sd4_validation_same_domain.txt" \
                  --t_dset_txt "data/sperm/sd4/same_domain/sd4_target_same_domain.txt" \
                  --no_of_classes 2 --output_dir "experiments" --gpu_id "0" --arch Xception\
                  --target_labelled true


