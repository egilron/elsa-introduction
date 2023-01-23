
# python3 run_ner.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name conll2003 \
#   --output_dir /tmp/test-ner \
#   --do_train \
#   --do_eval



  # python3 run_ner_er.py \
  # --model_name_or_path xlm-roberta-base \
  # --dataset_name "/home/egil/gits/tsa22/data/ds_elsa" \
  # --task_name elsa \
  # --output_dir /tmp/xlmr-elsa \
  # --overwrite_cache True \
  # --do_train \
  # --num_train_epochs 8 \
  # --do_eval \
  # --return_entity_level_metrics

  # xlm-roberta-base : xlmr-elsa
  # NbAiLab/roberta_jan_128_scandinavian : OSError: NbAiLab/roberta_jan_128_scandinavian does not appear to have a file named pytorch_model.bin but there is a file for Flax weights. Use `from_flax=True` to load this model from those weights.

model_var=xlm-roberta-base
model_var='/fp/projects01/ec30/models/norbert3-base'
model_var='/fp/projects01/ec30/models/nb-bert-base'
model_var='/fp/projects01/ec30/models/xlm-roberta-base'
model_var='/fp/projects01/ec30/models/flan-t5-base'
model_var='/fp/projects01/ec30/Cross_lingual_retrieval/models/norbert2'
model_var='/fp/projects01/ec30/Cross_lingual_retrieval/BRENT_Reader'
dataset_var='/fp/homes01/u01/ec-egilron/elsa-introduction/data/ds_elsa'

# Er visst viktig å bruke absolutt path
echo "xlmr"
echo $(date)

python3 run_sq_label.py \
  --model_name_or_path '/fp/projects01/ec30/models/xlm-roberta-base' \
  --tokenizer_name '/fp/projects01/ec30/models/xlm-roberta-base' \
  --dataset_name $dataset_var \
  --per_device_train_batch_size 32 \
  --task_name tsa-fox \
  --output_dir '/fp/projects01/ec30/egilron/elsa-inro/xlmr' \
  --overwrite_cache True \
  --do_train \
  --num_train_epochs 8 \
  --do_eval \
  --return_entity_level_metrics \
  --use_auth_token False \
  --label_column_name "tsa_tags"\
  --save_strategy epoch \
  --logging_strategy epoch \
  --do_predict \
  --log_level info \
  --logging_dir '/fp/projects01/ec30/egilron/elsa-intro/xlmr/logs'

echo "BRENT"
echo $(date)
python3 run_sq_label.py \
  --model_name_or_path '/fp/projects01/ec30/Cross_lingual_retrieval/BRENT_Reader' \
  --tokenizer_name '/fp/projects01/ec30/Cross_lingual_retrieval/models/norbert2' \
  --dataset_name $dataset_var \
  --per_device_train_batch_size 32 \
  --task_name tsa-fox \
  --output_dir '/fp/projects01/ec30/egilron/elsa-intro/brent' \
  --overwrite_cache True \
  --do_train \
  --num_train_epochs 8 \
  --do_eval \
  --return_entity_level_metrics \
  --use_auth_token False \
  --label_column_name "tsa_tags"\
  --text_column_name "tokens"
  --save_strategy epoch \
  --logging_strategy epoch \
  --do_predict  \
  --log_level info 


  echo "norBERT2"
echo $(date)
  python3 run_sq_label.py \
  --model_name_or_path '/fp/projects01/ec30/Cross_lingual_retrieval/models/norbert2' \
  --tokenizer_name '/fp/projects01/ec30/Cross_lingual_retrieval/models/norbert2' \
  --dataset_name $dataset_var \
  --per_device_train_batch_size 32 \
  --task_name tsa-fox \
  --output_dir '/fp/projects01/ec30/egilron/elsa-intro/norbert2' \
  --overwrite_cache True \
  --do_train \
  --num_train_epochs 8 \
  --do_eval \
  --return_entity_level_metrics \
  --use_auth_token False \
  --label_column_name "tsa_tags"\
  --save_strategy epoch \
  --logging_strategy epoch \
  --do_predict  \
  --text_column_name "tokens"\
  --log_level info 