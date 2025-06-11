while getopts ':c:l:a:s:b:w:d:v:p:e:t:y:x:' opt
do
    case $opt in
        c) CUDA_IDS="$OPTARG" ;;
        l) learning_rate="$OPTARG" ;;
        s) max_steps="$OPTARG" ;;
        b) subname="$OPTARG" ;;
        a) accumulate_grad_batches="$OPTARG" ;;
        w) warmup_steps="$OPTARG" ;;
        d) weight_decay="$OPTARG" ;;
        v) model_version="$OPTARG";;
        p) data_dir="$OPTARG";;
        e) base_data_dir="$OPTARG";;
        t) test_file_dir="$OPTARG";;
        y) test_size="$OPTARG";;
        x) data="$OPTARG";;
        ?)
        exit 1;;
    esac
done

if [ ! "${CUDA_IDS}" ]
then
  CUDA_IDS=0,1,2,3
fi


if [ ! "${learning_rate}" ]
then
  learning_rate=300
fi


if [ ! "${max_steps}" ]
then
  max_steps=2_000
fi


if [ ! "${subname}" ]
then
  subname=4m2d-seq2seq-40k-analysis-bf16
fi


if [ ! "${accumulate_grad_batches}" ]
then
  accumulate_grad_batches=5
fi


if [ ! "${warmup_steps}" ]
then
  warmup_steps=0
fi


if [ ! "${weight_decay}" ]
then
  weight_decay=0
fi


precision=bf16-mixed
gradient_clip_val=1

train_batch_size=20
eval_batch_size=80

max_seq_length=128
max_seq_length_output=400

model_name_or_path="./pretrained_models/t5-base"


current_date=$(date +"%Y-%m-%d")
cache_dir="./.cache/${current_date}/${subname}/"
output_dir="./output_model/${subname},lr=${learning_rate},ag=${accumulate_grad_batches},max_steps=${max_steps},warmup_steps=${warmup_steps},weight_decay=${weight_decay}"

echo "Ouput Model path : ${output_dir}/model/final_model version: ${model_version}"
val_check_interval=2000
val_check_interval=$(($val_check_interval * $accumulate_grad_batches))



if [[ $data == "Chatgpt" ]]; then
    datamodule=Seq2seqDataModuleChatgpt
elif [[ $data == "Llama" ]]; then
    datamodule=Seq2seqDataModuleLlama
else
    datamodule=Seq2seqDataModuleMixtral
fi


CUDA_VISIBLE_DEVICES=${CUDA_IDS} python seq2seq.py fit \
  --data ${datamodule} \
  --seed_everything 42 \
  --trainer.devices=1 \
  --trainer.accelerator=gpu \
  --trainer.enable_checkpointing=False \
  --trainer.precision=${precision} \
  --trainer.check_val_every_n_epoch null \
  --trainer.accumulate_grad_batches ${accumulate_grad_batches} \
  --trainer.num_sanity_val_steps 5 \
  --trainer.max_steps ${max_steps} \
  --trainer.val_check_interval ${val_check_interval} \
  --trainer.gradient_clip_val ${gradient_clip_val} \
  --data.base_data_dir "${base_data_dir}" \
  --data.data_dirs "${data_dir}" \
  --data.cache_dir "${cache_dir}" \
  --data.model_name_or_path "${model_name_or_path}" \
  --data.num_workers 12 \
  --data.test_file_dir "${test_file_dir}" \
  --data.test_size ${test_size} \
  --data.train_batch_size ${train_batch_size} \
  --data.eval_batch_size ${eval_batch_size} \
  --data.max_seq_length ${max_seq_length} \
  --data.max_seq_length_output ${max_seq_length_output} \
  --model.learning_rate ${learning_rate}e-5 \
  --model.warmup_steps ${warmup_steps} \
  --model.output_dir "${output_dir}" \
  --model.weight_decay ${weight_decay} \
  --model.model_name_or_path "${model_name_or_path}"

