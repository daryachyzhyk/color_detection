ssh -i ~/.ssh/id_rsa_jkobe -N -L 5433:db-catalog-back.dev.envs.lookiero.tech:5432 ubuntu@vpc-support.lookiero.tech &

PATH=/home/ubuntu/anaconda3/envs/amy3/bin
export PYTHONPATH=${PYTHONPATH}:/usr/share/lookiero/color_detection:/home/ubuntu/anaconda3/envs/amy3/lib/python3.7/site-packages/ds-data-core/
cd /usr/share/lookiero/color_detection/color_combination_model/

# python -c 'import color_extraction_with_mask_cron; color_extraction_with_mask_cron.color_detection_cron(local=False)'
python color_extraction_with_mask_cron.py