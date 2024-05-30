# Skeleton-based Action Recognition Using CTR-GCN with Angular Motion

- To preprocess the raw NTU-RGB+D data, which is available [here](https://github.com/shahroudy/NTURGB-D?tab=readme-ov-file), run:
```bash
python preprocess.py --data-dir <path-to-raw-data> --save-dir <where-to-save-data> --missing-files resources/ntu_missing.txt
```
- To train and evaluation models, use main.py. To see help message, run:
```bash
python main.py -h
```
- To evaluate dataset using multi models, run:
```bash
python main.py --data-path <path-to-data> --ensemble <models-to-use> --features <features-to-uses> --alphas <weight-of-each-model> --adaptive
```
    - Note that ensemble models, features and alphas must follow the same order

- To visualize/predict single sample, run
```bash
python visualize.py --sample-path <path-to-sample> --ensemble <models-to-use> --features <features-to-uses> --alphas <weight-of-each-model>--adaptive
```
- Checkpoints are available at [Google Drive](https://drive.google.com/drive/folders/1kJBfLjJ_fsmPCvDCAVG4iC2jS0Ezwsck?usp=sharing)
- Link to our notebook: [Colab Notebook](https://colab.research.google.com/drive/1vHozxI-9t9RHnxhSfvbJ6MIJ58CBiZSO?usp=sharing)
