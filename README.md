
# Medical Domain Adaptive Neural Networks
This is the Pytorch implementation for our paper [Adaptive Adversarial Neural Networks for Lossy and Domain-ShiftedMedical Image Analysis](http://shafieelab.bwh.harvard.edu). 

## Requirements
- Python 3.5
- Pytorch 1.4.0
- PyYAML 5.3.1
- scikit-image 0.14.0
- scikit-learn 0.20.0
- SciPy 1.1.0
- opencv-python 4.2.0.34
- Matplotlib 3.0.0
- NumPy 1.15.2
- TensorFlow 1.10.0 (For TensorBoard)

## Dataset
.txt files are lists for source and target domains 

The Embryo, Malaria, Sperm datasets are available online  [here](https://osf.io/dev35/). Once they are downloaded and extracted into your data directory, create .TXT files with filepaths and nummeric annotation with space delimited.

## Training

You can run MD-nets as follows
```
python -u experiments/tools/train_md_nets.py --mode train \  
              --seed $seed 
              --num_iterations 100000 --patience 5000 --test_interval 500 --snapshot_interval 1000 \  
              --dset embryo --s_dset ed4 --t_dset $target \  
              --s_dset_txt "data/embryo/ed4/ed4_source.txt" 
              --sv_dset_txt "data/embryo/ed4/ed4_validation.txt" \  
              --t_dset_txt "data/embryo/${target}/${target}_target.txt" \  
              --no_of_classes 5 \
              --output_dir "experiments" \
              --gpu_id 1 \
              --arch Xception\  
              --crop_size 224 --image_size 256
              --target_labelled true \  
              --trained_model_path ""
```

To run the experiments reported in the paper
```
./experiments/scripts/run_DATASETNAME_experiments.sh 
```

The experiment log file and the saved models will be stored at ./experiments/logs/experiment_name/ and ./experiments/models/experiment_name/
## Test

You can test the datasets on reported models as follows
```
./experiments/scripts/test_DATASETNAME.sh 
```


## Citing 
Please cite our paper if you use our code in your research:
```
@inproceedings{,
  title={},
  author={},
  booktitle={},
  pages={},
  year={}
}
```
## Contact
If you have any questions, please contact us via hshafiee[at]bwh.harvard.edu.
