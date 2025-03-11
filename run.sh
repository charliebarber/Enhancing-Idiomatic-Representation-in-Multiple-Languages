#!/bin/bash

# Define the pre-trained model to use
model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2" 

# Set GPU device (if you have multiple GPUs)
device_name='0'
export CUDA_VISIBLE_DEVICES=$device_name

# Hyperparameters to experiment with
for triplet_margin in 0.3
do
   for miner_margin in 0.4 
   do
      # Model configurations
      infoce=0.04
      maxlen=512 
      bs=128 # originally 64, 128 work on 3090 Ti
      dropout=0.05
      rs=33
      
      # Data path - update this to your data location
      data="./train_data/best_data_trainer.csv"
      
      # TRAIN MODEL
      # Use your system's Python interpreter
      python ./train_scripts/train.py \
         --model_name ${model} \
         --train_dir "${data}" \
         --output_dir ./model/${model}/$device_name \
         --epoch 24 \
         --train_batch_size $bs \
         --learning_rate 2e-5 \
         --max_length ${maxlen} \
         --parallel \
         --random_seed ${rs} \
         --loss "triplet_loss" \
         --infoNCE_tau ${infoce} \
         --dropout_rate ${dropout} \
         --agg_mode 'tokenmarker2layer' \
         --miner_margin $miner_margin \
         --type_of_triplets 'all' \
         --triplet_margin $triplet_margin \
         --training_mode "pre_training" \
         --use_miner \
         --device_name $device_name \
         --train_model \
         --add_idoms_to_tokenizer

      # DONT TRAIN MODEL, JUST EVALUATE
      # python ./train_scripts/train.py \
      #    --model_name ${model} \
      #    --train_dir "${data}" \
      #    --output_dir ./model/${model}/$device_name \
      #    --epoch 24 \
      #    --train_batch_size $bs \
      #    --learning_rate 2e-5 \
      #    --max_length ${maxlen} \
      #    --parallel \
      #    --random_seed ${rs} \
      #    --loss "triplet_loss" \
      #    --infoNCE_tau ${infoce} \
      #    --dropout_rate ${dropout} \
      #    --agg_mode 'tokenmarker2layer' \
      #    --miner_margin $miner_margin \
      #    --type_of_triplets 'all' \
      #    --triplet_margin $triplet_margin \
      #    --training_mode "pre_training" \
      #    --use_miner \
      #    --device_name $device_name \
      #    --add_idoms_to_tokenizer \
      #    --evaluate_trained_model
   done
done