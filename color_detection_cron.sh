PATH=/home/ubuntu/anaconda3/envs/amy3/bin
#PATH=/home/jkobe/anaconda3/envs/amy3/bin
export PYTHONPATH=${PYTHONPATH}:/usr/share/lookiero/color_detection:/home/ubuntu/anaconda3/envs/amy3/lib/python3.7/site-packages/ds-data-core/
#export PYTHONPATH=${PYTHONPATH}:/home/jkobe/Lookiero/color_detection:/home/jkobe/anaconda3/envs/amy3/lib/python3.7/site-packages/ds-data-core/
cd /usr/share/lookiero/color_detection/
#cd /home/jkobe/Lookiero/color_detection/

python -c 'import color_detection_cron; color_detection_cron.color_detection_cron(local=False)'
