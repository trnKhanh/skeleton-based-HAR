# Skeleton-based Action Recognition Using CTR-GCN with Angular Motion

- To preprocess the raw NTU-RGB+D data, which is available [here](https://github.com/shahroudy/NTURGB-D?tab=readme-ov-file), run:
```bash
python preprocess.py --data-dir <path-to-raw-data> --save-dir <where-to-save-data> --missing-files resources/ntu_missing.txt
```
- To train and evaluation models, use main.py. To see help message, run:
```bash
python main.py -h
```
- To visualize/predict single sample, run
```bash
python visualize.py --sample-path <path-to-sample> --ensemble <models-to-use> --features <features-to-uses> --adaptive
```
- Checkpoints are available at [Google Drive](https://drive.google.com/drive/folders/1kJBfLjJ_fsmPCvDCAVG4iC2jS0Ezwsck?usp=sharing)
