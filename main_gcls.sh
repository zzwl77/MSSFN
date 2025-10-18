dataset="/media/mprl2/Hard Disk/zwl/3method_trans/gazedata240828"
for i in $(seq 1 100)
do
  for questionnaire in "cls" 
  do
    for n_fold in 0 1 2 3 4
    do
      python main_gcls.py \
      --dataset "$dataset" \
      --datacsv "adnc_10f4" \
      --questionnaire "$questionnaire" \
      --save_file "adghv_n.pt" \
      --n_fold "$n_fold"  \
      --train-batch 20 \
      --test-batch 20  \
      --gpu_id "0"  \
      --arch adg6 \
      --epochs 100 \
      --patience 20  \
      --best_acc 0.8
    done
  done
done