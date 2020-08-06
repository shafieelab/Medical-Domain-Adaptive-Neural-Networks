seed=0
# run sperm unlabelled settings
# SD4 to SD3, SD2 and SD4 adaption
printf "sd4 to sd4 slides \n \n"

python -u experiments/tools/train_md_nets.py --mode test \
              --seed $seed --dset sperm --s_dset sd4 --t_dset sd4_slides \
              --s_dset_txt "data/sperm/sd4/sd4_source.txt" --sv_dset_txt "data/sperm/sd4/sd4_validation.txt" \
              --t_dset_txt "data/sperm/sd4/sd4_patient_target.txt" \
              --no_of_classes 2 --output_dir "experiments"  --gpu_id "0" --arch Xception\
              --target_labelled false --crop_size 256 --image_size 224 \
              --sperm_patient_data_clinicians_annotations 'data/sperm/sperm_patient_data_clinicians_annotations.csv'\
          --trained_model_path "experiments/submitted_models/sperm/Xception/train_sd4_to_sd4_slides/best_model.pth.tar"

printf "DONE \n \n \n "


printf "sd4 to sd3 slides \n \n"

python -u experiments/tools/train_md_nets.py --mode test \
              --seed $seed --dset sperm --s_dset sd4 --t_dset sd3_slides \
              --s_dset_txt "data/sperm/sd4/sd4_source.txt" --sv_dset_txt "data/sperm/sd4/sd4_validation.txt" \
              --t_dset_txt "data/sperm/sd3/sd3_patient_target.txt" \
              --no_of_classes 2 --output_dir "experiments"  --gpu_id "0" --arch Xception\
              --target_labelled false --crop_size 256 --image_size 224 \
              --sperm_patient_data_clinicians_annotations 'data/sperm/sperm_patient_data_clinicians_annotations.csv'\
          --trained_model_path "experiments/submitted_models/sperm/Xception/train_sd4_to_sd3_slides/best_model.pth.tar"

printf "DONE \n \n \n "
printf "sd4 to sd2 slides \n \n"

python -u experiments/tools/train_md_nets.py --mode test \
              --seed $seed --dset sperm --s_dset sd4 --t_dset sd2_slides \
              --s_dset_txt "data/sperm/sd4/sd4_source.txt" --sv_dset_txt "data/sperm/sd4/sd4_validation.txt" \
              --t_dset_txt "data/sperm/sd2/sd2_patient_target.txt" \
              --no_of_classes 2 --output_dir "experiments"  --gpu_id "0" --arch Xception\
              --target_labelled false --crop_size 280 --image_size 280 \
              --sperm_patient_data_clinicians_annotations 'data/sperm/sperm_patient_data_clinicians_annotations.csv'\
          --trained_model_path "experiments/submitted_models/sperm/Xception/train_sd4_to_sd2_slides/best_model.pth.tar"


printf "DONE \n \n \n "
printf "sd4 to sd1 slides \n \n"

python -u experiments/tools/train_md_nets.py --mode test \
              --seed $seed --dset sperm --s_dset sd4 --t_dset sd1_slides \
              --s_dset_txt "data/sperm/sd4/sd4_source.txt" --sv_dset_txt "data/sperm/sd4/sd4_validation.txt" \
              --t_dset_txt "data/sperm/sd1/sd1_patient_target.txt" \
              --no_of_classes 2 --output_dir "experiments"  --gpu_id "0" --arch Xception\
              --target_labelled false --crop_size 130 --image_size 130 \
              --sperm_patient_data_clinicians_annotations 'data/sperm/sperm_patient_data_clinicians_annotations.csv'\
          --trained_model_path "experiments/submitted_models/sperm/Xception/train_sd4_to_sd1_slides/best_model.pth.tar"

printf "DONE \n \n \n "


# run sperm labelled settings
# SD4 to SD4 adaption
printf "SD4 to SD4 Labelled \n \n"

python -u experiments/tools/train_md_nets.py --mode test \
                  --seed $seed --dset sperm  --s_dset sd4 --t_dset sd4 \
                  --s_dset_txt "data/sperm/sd4/same_domain/sd4_source_same_domain.txt" \
                  --sv_dset_txt "data/sperm/sd4/same_domain/sd4_validation_same_domain.txt" \
                  --t_dset_txt "data/sperm/sd4/same_domain/sd4_target_same_domain.txt" \
                  --no_of_classes 2 --output_dir "experiments" --gpu_id "0" --arch Xception\
                  --target_labelled true --crop_size 256 --image_size 224 \
              --sperm_patient_data_clinicians_annotations 'data/sperm/sperm_patient_data_clinicians_annotations.csv'\
          --trained_model_path "experiments/submitted_models/sperm/Xception/train_sd4_to_sd4/best_model.pth.tar"


