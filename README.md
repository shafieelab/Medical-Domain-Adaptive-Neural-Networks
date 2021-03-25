
# Medical Domain Adaptive Neural Networks
This is the Pytorch implementation for our paper [Adaptive Adversarial Neural Networks for Lossy and Domain-Shifted Medical Image Analysis](http://shafieelab.bwh.harvard.edu). 

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

## Framework 
### MD-nets

### MD-nets(nos) without source data

## Dataset
.txt files are lists for source and target domains 

The Embryo, Malaria, Sperm  are available online  [here](https://osf.io/dev35/) and Office-31 datasets from [here](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code)  Once they are downloaded and extracted into your data directory, create .TXT files with filepaths and numeric annotation with space delimited.

## Training

You can train MD-nets as follows
```
python -u experiments/python_scripts/train_md_nets.py --mode train \  
              --seed $seed 
              --num_iterations 100000 --patience 5000 --test_interval 500 --snapshot_interval 1000 \  
              --dset dataset_name --s_dset source_name --t_dset target_name \  
              --lr "0.0001"  --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256
              --s_dset_txt "source_training.txt" 
              --sv_dset_txt "source_validation.txt" \  
              --t_dset_txt "target.txt" \  
              --no_of_classes 5 \
              --output_dir "experiments" \
              --gpu_id 1 \
              --arch Xception\  
              --crop_size 224 --image_size 256
              --target_labelled true  --trade_off 1.0 \  
              --trained_model_path ""
```
##### For MD-nets (Nos) as follows
```
python -u experiments/python_scripts/train_md_nets_nos.py --mode train \  
              --seed $seed 
              --num_iterations 10000 --patience 5000 --test_interval 500 --snapshot_interval 1000 \  
              --dset embryo --s_dset source_name --t_dset target_name \  
              --lr "0.0001"  --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256
              --tv_dset_txt "target_validation.txt" \  
              --t_dset_txt "target.txt" \  
              --no_of_classes 31 \
              --output_dir "experiments" \
              --gpu_id 0 \
              --arch ResNet50\  
              --crop_size 224 --image_size 256
              --source_model_path "model.pth.tar"
              --trade_off_cls 0.3 
```


To run the experiments reported in the paper
```
./experiments/scripts/run_DATASETNAME_experiments.sh 
```

The experiment log file and the saved models will be stored at ./experiments/logs/experiment_name/ and ./experiments/models/experiment_name/
## Testing

You can test the datasets on reported models as follows
```
./experiments/scripts/test_DATASETNAME.sh 
```

<!---
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
-->
## Contact

If you have any questions, please contact us via hshafiee[at]bwh.harvard.edu.
