import numpy as np
import os
import glob

video_id = 'chat_response'
processed_dir = f'data/processed/videos/{video_id}'
coeff_path = os.path.join(processed_dir, 'vid_coeff.npy')
ori_imgs_dir = os.path.join(processed_dir, 'ori_imgs')

print(f"Loading {coeff_path}...")
data = np.load(coeff_path, allow_pickle=True).item()
lm68_arr = data['lm68'] # [N, 68, 2]

print(f"Found {len(lm68_arr)} frames of landmarks.")

image_paths = sorted(glob.glob(os.path.join(ori_imgs_dir, '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
print(f"Found {len(image_paths)} images.")

if len(image_paths) != len(lm68_arr):
    print(f"Warning: Mismatch in frames. Images: {len(image_paths)}, Landmarks: {len(lm68_arr)}")

for i, image_path in enumerate(image_paths):
    if i >= len(lm68_arr): break
    lms = lm68_arr[i]
    np.savetxt(image_path.replace('.jpg', '.lms'), lms, '%f')

print("Converted npy landmarks to .lms files.")
