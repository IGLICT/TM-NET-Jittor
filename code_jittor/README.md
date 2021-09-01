# Car
python code/extract_latents_central_part.py \
--image_dir /mnt/f/wutong/car/ \
--mat_dir /mnt/f/wutong/car \
--vqvae_ckpt ./20210401_car/latest.pth \
--vqvae_yaml ./code/yaml/train_vqvae.yml \
--geovae_ckpt ./20210401_car_geo/latest.pth \
--geovae_yaml ./code/yaml/train_geovae.yml \
--category car \
--save_path ./car_central \
--device 3 \
--mode 'train'

python code/conditional_sample_2levels_central_part.py \
    --path ./car_latents/ \
    --part_name body \
    --vqvae ./20210413_car_new_reg/latest1.pth \
    --vqvae_yaml ./code/yaml/car_new_reg/train_vqvae.yml \
    --top ./20210414_car_new_reg_pixelsnail/top_8/latest.pth \
    --top_yaml ./code/yaml/car_new_reg/train_pixelsnail_top_center_8.yml \
    --bottom ./20210414_car_new_reg_pixelsnail/bottom/latest.pth \
    --bottom_yaml ./code/yaml/car_new_reg/train_pixelsnail_bottom_center.yml \
    --device 2 \
    --batch 2

python code/conditional_sample_2levels_central_part.py --path ./car_latents/ --part_name body --vqvae ./share/latest_bak_800.pth --vqvae_yaml ./code/yaml/car/train_vqvae.yml --top ./share/latest.pth --top_yaml ./code/yaml/car/train_pixelsnail_top_center_8.yml --bottom ./20210414_car_pixelsnail/bottom/latest.pth --bottom_yaml ./code/yaml/car/train_pixelsnail_bottom_center.yml --device 2 --batch 2

python code/extract_latents_central_part_aug.py \
--image_dir /mnt/f/wutong/car/ \
--mat_dir /mnt/f/wutong/car \
--vqvae_ckpt ./20210416_car_aug/latest1.pth \
--vqvae_yaml ./code/yaml/car_aug/train_vqvae.yml \
--geovae_ckpt ./20210401_car_geo/latest_bak_200.pth \
--geovae_yaml ./code/yaml/car/train_geovae.yml \
--category car \
--save_path ./20210417_car_aug_latents \
--device 0 \
--mode 'train'

python code/conditional_sample_2levels_central_part.py \
--path ./20210417_car_aug_latents \
--part_name body \
--vqvae ./20210416_car_aug/latest1.pth \
--vqvae_yaml ./code/yaml/car_aug/train_vqvae.yml \
--top ./20210417_car_aug_pixelsnail/top_16/latest.pth \
--top_yaml ./code/yaml/car_aug/train_pixelsnail_top_center_16.yml \
--bottom ./20210417_car_aug_pixelsnail/bottom/latest.pth \
--bottom_yaml ./code/yaml/car_aug/train_pixelsnail_bottom_center.yml \
--device 0 \
--batch 1

# Car_new_reg
python code/extract_latents_central_part.py \
    --image_dir /mnt/f/wutong/car/ \
    --mat_dir /mnt/f/wutong/car_new_reg \
    --vqvae_ckpt ./20210413_car_new_reg/latest1.pth \
    --vqvae_yaml ./code/yaml/car_new_reg/train_vqvae.yml \
    --geovae_ckpt ./20210413_car_new_reg_geo/latest1.pth \
    --geovae_yaml ./code/yaml/car_new_reg/train_geovae.yml \
    --category car \
    --save_path ./20210414_car_new_reg_latents \
    --device 0 \
    --mode 'train'

python code/extract_latents_central_part_aug.py \
--image_dir /mnt/f/wutong/data/car_new_reg/ \
--mat_dir /mnt/f/wutong/data/car_new_reg \
--vqvae_ckpt ./20210417_car_new_reg_aug/latest1.pth \
--vqvae_yaml ./code/yaml/car_new_reg_aug/train_vqvae.yml \
--geovae_ckpt ./20210413_car_new_reg_geo/latest1.pth \
--geovae_yaml ./code/yaml/car_new_reg/train_geovae.yml \
--category car \
--save_path /mnt/f/wutong/latent_data/20210417_car_new_reg_aug_latents \
--device 2 \
--mode 'train'

python code/conditional_sample_2levels_central_part.py \
--path /mnt/f/wutong/latent_data/20210417_car_new_reg_aug_latents \
--part_name body \
--vqvae ./20210417_car_new_reg_aug/latest1.pth \
--vqvae_yaml ./code/yaml/car_new_reg_aug/train_vqvae.yml \
--top ./20210417_car_new_reg_aug_pixelsnail/top_16/latest.pth \
--top_yaml ./code/yaml/car_new_reg_aug/train_pixelsnail_top_center_16.yml \
--bottom ./20210417_car_new_reg_aug_pixelsnail/bottom/latest.pth \
--bottom_yaml ./code/yaml/car_new_reg_aug/train_pixelsnail_bottom_center.yml \
--device 0 \
--batch 1

python -m pdb code/extract_latents_central_part.py \
--image_dir /mnt/f/wutong/data/car_new_reg \
--mat_dir /mnt/f/wutong/data/car_new_reg \
--vqvae_ckpt ./20210417_car_new_reg_aug/latest1.pth \
--vqvae_yaml ./code/yaml/car_new_reg_aug/train_vqvae.yml \
--geovae_ckpt ./20210413_car_new_reg_geo/wheel/latest1.pth \
--geovae_yaml ./code/yaml/car_new_reg_aug/train_geovae.yml \
--category car \
--save_path /mnt/f/wutong/latent_data/20210417_car_new_reg_aug_latents \
--device 1 \
--mode 'val'

python code/conditional_sample_2levels_central_part.py \
--path /mnt/f/wutong/latent_data/20210417_car_new_reg_aug_latents \
--part_name right_front_wheel \
--vqvae ./20210417_car_new_reg_aug/latest1.pth \
--vqvae_yaml ./code/yaml/car_new_reg_aug/right_front_wheel/train_vqvae.yml \
--top ./20210417_car_new_reg_aug_pixelsnail/right_front_wheel/top_16/latest.pth \
--top_yaml ./code/yaml/car_new_reg_aug/right_front_wheel/train_pixelsnail_top_center_16.yml \
--bottom ./20210417_car_new_reg_aug_pixelsnail/right_front_wheel/bottom/latest.pth \
--bottom_yaml ./code/yaml/car_new_reg_aug/right_front_wheel/train_pixelsnail_bottom_center.yml \
--device 0 \
--batch 1

# Chair
python code/extract_latents_central_part.py \
--image_dir /mnt/f/wutong/chair/ \
--mat_dir /mnt/f/wutong/chair \
--vqvae_ckpt ./20210412_chair/latest1.pth \
--vqvae_yaml ./code/yaml/chair/train_vqvae.yml \
--geovae_ckpt ./20210412_chair_geo/latest1.pth \
--geovae_yaml ./code/yaml/chair/train_geovae.yml \
--category chair \
--save_path ./20210412_chair_latent \
--device 0 \
--mode 'train'

python code/conditional_sample_2levels_central_part.py \
--path ./20210412_chair_latent \
--part_name back \
--vqvae ./20210412_chair/latest1.pth \
--vqvae_yaml ./code/yaml/chair/train_vqvae.yml \
--top ./20210412_chair_pixelsnail/top_16/latest.pth \
--top_yaml ./code/yaml/chair/train_pixelsnail_top_center_16.yml \
--bottom ./20210412_chair_pixelsnail/bottom/latest.pth \
--bottom_yaml ./code/yaml/chair/train_pixelsnail_bottom_center.yml \
--device 0 \
--batch 1

python -m pdb code/extract_latents_other_parts.py \
--image_dir /mnt/f/wutong/data/chair \
--mat_dir /mnt/f/wutong/data/chair \
--vqvae_ckpt ./20210412_chair/latest1.pth \
--vqvae_yaml ./code/yaml/chair/train_vqvae.yml \
--geovae_ckpt_dir ./20210412_chair_geo \
--geovae_yaml ./code/yaml/chair/train_geovae.yml \
--category chair \
--save_path /mnt/f/wutong/latent_data/20210412_chair_latents_others \
--device 3 \
--mode 'val'

python code/conditional_sample_2levels_other_parts.py \
--path ../../../205_f/wutong/latent_data/20210516_chair_latents_others \
--central_part_name surface \
--part_name leg \
--vqvae ./20210412_chair/latest1.pth \
--vqvae_yaml ./code/yaml/chair/train_vqvae.yml \
--top ./20210412_chair_pixelsnail/leg/top_16/latest.pth \
--top_yaml ./code/yaml/chair/leg/train_pixelsnail_top_center_16.yml \
--bottom ./20210412_chair_pixelsnail/leg/bottom/latest.pth \
--bottom_yaml ./code/yaml/chair/leg/train_pixelsnail_bottom_center.yml \
--central_part_sample_dir ./20210412_chair_pixelsnail/top_16/auto_texture \
--device 0 \
--batch 1

# Table
python code/extract_latents_geo_only_all_parts.py \
--mat_dir /mnt/f/wutong/table \
--category table \
--vertex_num 4332 \
--geovae_yaml ./code/yaml/table/train_geovae.yml \
--geovae_ckpt_dir ./20210422_table_geo \
--save_path ./20210422_table_latents \
--device 0 \
--mode 'train'

python code_jittor/extract_latents_central_part.py \
--image_dir /mnt/f/wutong/data/table/ \
--mat_dir /mnt/f/wutong/data/table \
--vqvae_ckpt ./20210422_table/latest1.pkl \
--vqvae_yaml ./code_jittor/yaml/table/surface/train_vqvae.yml \
--geovae_ckpt ./20210422_table_geo/surface/latest1.pkl \
--geovae_yaml ./code_jittor/yaml/table/surface/train_geovae.yml \
--category table \
--save_path ./20210422_table_latents \
--device 0 \
--mode 'train'

python code_jittor/conditional_sample_2levels_central_part.py \
--path ./20210422_table_latents \
--part_name surface \
--vqvae ./20210422_table/latest1.pkl \
--vqvae_yaml ./code_jittor/yaml/table/surface/train_vqvae.yml \
--top ./20210422_table_pixelsnail/top_16/latest.pkl \
--top_yaml ./code_jittor/yaml/table/surface/train_pixelsnail_top_center_16.yml \
--bottom ./20210422_table_pixelsnail/bottom/latest.pkl \
--bottom_yaml ./code_jittor/yaml/table/surface/train_pixelsnail_bottom_center.yml \
--device 0 \
--batch 1

python -m pdb code_jittor/extract_latents_other_parts.py \
--image_dir /mnt/f/wutong/data/table/ \
--mat_dir /mnt/f/wutong/data/table \
--vqvae_ckpt ./20210422_table/latest1.pkl \
--vqvae_yaml ./code_jittor/yaml/table/leg/train_vqvae.yml \
--geovae_ckpt_dir ./20210422_table_geo \
--geovae_yaml ./code_jittor/yaml/table/leg/train_geovae.yml \
--category table \
--save_path ./20210422_table_latents \
--device 3 \
--mode 'val'

python code_jittor/conditional_sample_2levels_other_parts.py \
--path ./20210522_table_latents \
--central_part_name surface \
--part_name leg \
--vqvae ./20210422_table/latest1.pkl \
--vqvae_yaml ./code_jittor/yaml/table/leg/train_vqvae.yml \
--top ./20210422_table_pixelsnail/leg/top_16/latest.pkl \
--top_yaml ./code_jittor/yaml/table/leg/train_pixelsnail_top_center_16.yml \
--bottom ./20210422_table_pixelsnail/leg/bottom/latest.pkl \
--bottom_yaml ./code_jittor/yaml/table/leg/train_pixelsnail_bottom_center.yml \
--central_part_sample_dir ./20210422_table_pixelsnail/surface/top_16/auto_texture \
--device 0 \
--batch 1
<!-- --path /mnt/f/wutong/latent_data/20210516_table_latents_others \ -->

# Plane
python code/extract_latents_central_part.py \
--image_dir /mnt/f/wutong/data/plane \
--mat_dir /mnt/f/wutong/data/plane \
--vqvae_ckpt ./20210425_plane/latest1.pth \
--vqvae_yaml ./code/yaml/plane/train_vqvae.yml \
--geovae_ckpt ./20210425_plane_geo/latest1.pth \
--geovae_yaml ./code/yaml/plane/train_geovae.yml \
--category plane \
--save_path /mnt/f/wutong/latent_data/20210425_plane_latents \
--device 0 \
--mode 'train'

python code/extract_latents_central_part.py \
--image_dir /mnt/f/wutong/data/plane \
--mat_dir /mnt/f/wutong/data/plane \
--vqvae_ckpt ./20210425_plane/body/latest1.pth \
--vqvae_yaml ./code/yaml/plane/train_vqvae.yml \
--geovae_ckpt ./20210425_plane_geo/body/latest1.pth \
--geovae_yaml ./code/yaml/plane/train_geovae.yml \
--category plane \
--save_path /mnt/f/wutong/latent_data/20210425_plane_latents \
--device 0 \
--mode 'train'

python code/conditional_sample_2levels_central_part.py \
--path /mnt/f/wutong/latent_data/20210425_plane_latents \
--part_name body \
--vqvae ./20210425_plane/body/latest1.pth \
--vqvae_yaml ./code/yaml/plane/train_vqvae.yml \
--top ./20210425_plane_pixelsnail/body/top_16/latest.pth \
--top_yaml ./code/yaml/plane/train_pixelsnail_top_center_16.yml \
--bottom ./20210425_plane_pixelsnail/body/bottom/latest.pth \
--bottom_yaml ./code/yaml/plane/train_pixelsnail_bottom_center.yml \
--device 0 \
--batch 1

python code/conditional_sample_2levels_central_part.py \
--path /mnt/f/wutong/latent_data/20210425_plane_latents \
--part_name up_tail \
--vqvae ./20210425_plane/latest1.pth \
--vqvae_yaml ./code/yaml/plane/train_vqvae.yml \
--top ./20210425_plane_pixelsnail/top_16/latest.pth \
--top_yaml ./code/yaml/plane/train_pixelsnail_top_center_16.yml \
--bottom ./20210425_plane_pixelsnail/bottom/latest.pth \
--bottom_yaml ./code/yaml/plane/train_pixelsnail_bottom_center.yml \
--device 0 \
--batch 1

python code/extract_latents_central_part_aug.py \
--image_dir /mnt/f/wutong/data/plane/ \
--mat_dir /mnt/f/wutong/data/plane \
--vqvae_ckpt ./20210425_plane/latest1.pth \
--vqvae_yaml ./code/yaml/plane/train_vqvae.yml \
--geovae_ckpt ./20210425_plane_geo/latest1.pth \
--geovae_yaml ./code/yaml/plane/train_geovae.yml \
--category plane \
--save_path /mnt/f/wutong/latent_data/20210425_plane_aug_latents \
--device 3 \
--mode 'train'

python code/extract_latents_central_part_aug.py \
--image_dir /mnt/f/wutong/data/plane/ \
--mat_dir /mnt/f/wutong/data/plane \
--vqvae_ckpt ./20210425_plane/body/latest1_aug.pth \
--vqvae_yaml ./code/yaml/plane/train_vqvae.yml \
--geovae_ckpt ./20210425_plane_geo/body/latest1.pth \
--geovae_yaml ./code/yaml/plane/train_geovae.yml \
--category plane \
--save_path /mnt/f/wutong/latent_data/20210425_plane_aug_latents \
--device 1 \
--mode 'test'


python code/conditional_sample_2levels_central_part.py \
--path ../../../205_f/wutong/latent_data/20210425_plane_aug_latents \
--part_name body \
--vqvae ./20210425_plane/body/latest1_aug.pth \
--vqvae_yaml ./code/yaml/plane_aug/train_vqvae.yml \
--top ./20210425_plane_aug_pixelsnail/body/top_16/latest.pth \
--top_yaml ./code/yaml/plane_aug/train_pixelsnail_top_center_16.yml \
--bottom ./20210425_plane_aug_pixelsnail/body/bottom/latest.pth \
--bottom_yaml ./code/yaml/plane_aug/train_pixelsnail_bottom_center.yml \
--device 1 \
--batch 1

