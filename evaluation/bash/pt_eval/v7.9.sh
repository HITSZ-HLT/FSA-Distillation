while getopts ':c:' opt
do
    case $opt in
        c) CUDA_IDS="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done

# Replace model version and subname as needed
model_version=v7.9
subname=5m16d-seq2seq-2000k-mixtral-200k-steps


max_steps=200000


# data = Mixtral or Chatgpt or Llama
data="Mixtral"

base_data_dir="./prompting/data/mixtral"


data_dir="rewrite/rewrite/restaurant/restaurant_20000_s12421_v0_bf16.json__\
rewrite/rewrite/computer/computer_20000_s12421_v0_bf16.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v0_bf16.json__\
analysis/analysis/computer/computer_20000_s12421_v0_bf16.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v1_bf16.json__\
rewrite/rewrite/computer/computer_20000_s12421_v1_bf16.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v1_bf16.json__\
analysis/analysis/computer/computer_20000_s12421_v1_bf16.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v2_bf16.json__\
rewrite/rewrite/computer/computer_20000_s12421_v2_bf16.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v2_bf16.json__\
analysis/analysis/computer/computer_20000_s12421_v2_bf16.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v3_bf16.json__\
rewrite/rewrite/computer/computer_20000_s12421_v3_bf16.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v3_bf16.json__\
analysis/analysis/computer/computer_20000_s12421_v3_bf16.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v4_bf16.json__\
rewrite/rewrite/computer/computer_20000_s12421_v4_bf16.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v4_bf16.json__\
analysis/analysis/computer/computer_20000_s12421_v4_bf16.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v5_bf16.json__\
rewrite/rewrite/computer/computer_20000_s12421_v5_bf16.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v5_bf16.json__\
analysis/analysis/computer/computer_20000_s12421_v5_bf16.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v6_bf16.json__\
rewrite/rewrite/computer/computer_20000_s12421_v6_bf16.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v6_bf16.json__\
analysis/analysis/computer/computer_20000_s12421_v6_bf16.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v7_bf16.json__\
rewrite/rewrite/computer/computer_20000_s12421_v7_bf16.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v7_bf16.json__\
analysis/analysis/computer/computer_20000_s12421_v7_bf16.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v8_int8.json__\
rewrite/rewrite/computer/computer_20000_s12421_v8_bf16.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v8_bf16.json__\
analysis/analysis/computer/computer_20000_s12421_v8_int8.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v9_int8.json__\
rewrite/rewrite/computer/computer_20000_s12421_v9_int8.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v9_int8.json__\
analysis/analysis/computer/computer_20000_s12421_v9_int8.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v10_int8.json__\
rewrite/rewrite/computer/computer_20000_s12421_v10_int8.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v10_int8.json__\
analysis/analysis/computer/computer_20000_s12421_v10_int8.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v11_int8.json__\
rewrite/rewrite/computer/computer_20000_s12421_v11_int8.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v11_int8.json__\
analysis/analysis/computer/computer_20000_s12421_v11_int8.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v12_int8.json__\
rewrite/rewrite/computer/computer_20000_s12421_v12_int8.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v12_int8.json__\
analysis/analysis/computer/computer_20000_s12421_v12_int8.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v13_int8.json__\
rewrite/rewrite/computer/computer_20000_s12421_v13_int8.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v13_int8.json__\
analysis/analysis/computer/computer_20000_s12421_v13_int8.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v14_int8.json__\
rewrite/rewrite/computer/computer_20000_s12421_v14_int8.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v14_int8.json__\
analysis/analysis/computer/computer_20000_s12421_v14_int8.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v15_int8.json__\
rewrite/rewrite/computer/computer_20000_s12421_v15_int8.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v15_int8.json__\
analysis/analysis/computer/computer_20000_s12421_v15_int8.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v16_int8.json__\
rewrite/rewrite/computer/computer_20000_s12421_v16_int8.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v16_int8.json__\
analysis/analysis/computer/computer_20000_s12421_v16_int8.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v17_int8.json__\
rewrite/rewrite/computer/computer_20000_s12421_v17_int8.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v17_int8.json__\
analysis/analysis/computer/computer_20000_s12421_v17_int8.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v18_int8.json__\
rewrite/rewrite/computer/computer_20000_s12421_v18_int8.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v18_int8.json__\
analysis/analysis/computer/computer_20000_s12421_v18_int8.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v19_int8.json__\
rewrite/rewrite/computer/computer_20000_s12421_v19_int8.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v19_int8.json__\
analysis/analysis/computer/computer_20000_s12421_v19_int8.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v20_int8.json__\
rewrite/rewrite/computer/computer_20000_s12421_v20_int8.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v20_int8.json__\
analysis/analysis/computer/computer_20000_s12421_v20_int8.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v21_int8.json__\
rewrite/rewrite/computer/computer_20000_s12421_v21_int8.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v21_int8.json__\
analysis/analysis/computer/computer_20000_s12421_v21_int8.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v22_int8.json__\
rewrite/rewrite/computer/computer_20000_s12421_v22_int8.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v22_int8.json__\
analysis/analysis/computer/computer_20000_s12421_v22_int8.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v23_int8.json__\
rewrite/rewrite/computer/computer_20000_s12421_v23_int8.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v23_int8.json__\
analysis/analysis/computer/computer_20000_s12421_v23_int8.json__\
rewrite/rewrite/restaurant/restaurant_20000_s12421_v24_int8.json__\
rewrite/rewrite/computer/computer_20000_s12421_v24_int8.json__\
analysis/analysis/restaurant/restaurant_20000_s12421_v24_int8.json__\
analysis/analysis/computer/computer_20000_s12421_v24_int8.json"


test_file_dir="./prompting/test/mixtral/test_500_4m8d.json"
test_size=500

# 蒸馏预训练

cd ./pre-training
chmod +x bash/3m13d/*
bash/3m13d/joint.sh -c ${CUDA_IDS} -s ${max_steps} -b ${subname} -v ${model_version} -p ${data_dir} -e ${base_data_dir} -t ${test_file_dir} -y ${test_size} -x ${data}
