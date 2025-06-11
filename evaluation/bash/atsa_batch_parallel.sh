while getopts ':c:m:d:b:s:l:p:n:' opt
do
    case $opt in
        c) CUDA_IDS="$OPTARG" ;;
        m) model_name_or_path="$OPTARG" ;;
        d) datasets="$OPTARG" ;;
        b) subname="$OPTARG" ;;
        s) seeds="$OPTARG" ;;
        l) learning_rate="$OPTARG" ;;
        p) data_prop="$OPTARG" ;;
        n) n_worker="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done



if [ -z "${seeds}" ]; then
    seeds="42 52 62 72 82 142 152 162 172 182"
fi

IFS=' ' read -r -a seed_array <<< "$seeds"



if [ -z "${datasets}" ]; then
    datasets="atsa/rest14 atsa/laptop14"
fi

IFS=' ' read -r -a dataset_array <<< "$datasets"



if [ -z "${model_name_or_path}" ]; then
    model_name_or_path='subname'
fi


if [ -z "${learning_rate}" ]; then
    learning_rate=30
fi


if [ -z "${data_prop}" ]; then
    data_prop=1.
fi


if [ -z "${n_worker}" ]; then
    n_worker=3
fi


parallel -j${n_worker} \
    bash/atsa.sh -d {1} -c ${CUDA_IDS} -b ${subname} -s {2} -m ${model_name_or_path} -l ${learning_rate} -p ${data_prop} \
    ::: ${dataset_array[@]} \
    ::: ${seed_array[@]}


# 串行
# for seed in "${seed_array[@]}"; do
#    for dataset in "${dataset_array[@]}"; do
#        echo "bash/atsa.sh -d ${dataset} -c ${CUDA_IDS} -b ${subname} -s ${seed} -m ${model_name_or_path} -l ${learning_rate} -p ${data_prop}"
#        bash/atsa.sh -d ${dataset} -c ${CUDA_IDS} -b ${subname} -s ${seed} -m ${model_name_or_path} -l ${learning_rate} -p ${data_prop}
#    done
# done

# bash/atsa.sh -d atsa/rest14 -c 1 -b base -s 42 -m subname -l 30 -p 1
# bash/atsa.sh -d atsa/laptop14 -c 1 -b base -s 42 -m subname -l 30 -p 1