#! /bin/sh

data_dir=/data/pos_sejong800k/s2s  # `pwd`
train_dir=/data/aan_train/pos_sejong800k_aan_slash_v1

export CUDA_VISIBLE_DEVICES=0

python code/trainer.py \
  --input $data_dir/pos_sejong800k.train.inputs.encoded $data_dir/pos_sejong800k.train.targets.encoded \
  --model transformer --output $train_dir \
  --vocabulary $data_dir/pos_sejong800k.vocab.inputs.cut $data_dir/pos_sejong800k.vocab.targets.cut \
  --validation $data_dir/pos_sejong800k.dev.inputs.encoded \
  --references $data_dir/pos_sejong800k.dev.targets.encoded \
  --parameters=batch_size=3125,device_list=[0],eval_steps=5000,train_steps=100000,save_checkpoint_steps=1500,shared_embedding_and_softmax_weights=true,shared_source_target_embedding=false,update_cycle=8,aan_mask=True,use_ffn=False
