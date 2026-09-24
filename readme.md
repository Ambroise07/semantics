# Semantics - A Text Generator

Semantics is a character-level text generator designed to synthesize sentences based on a training dataset of words. The generated sequences learn the underlying statistical patterns of the input data to produce text that resembles the original dataset.

This is my very first AI project, built after watching the amazing YouTube video  by Andrej Karpathy.
links: https://www.youtube.com/watch?v=TCH_1BHY58I

---

## Contact Me

* **Email:** ambroiseisrael5@gmail.com
* **Website:** *Coming soon*

---

## Running the Model

Assuming you have a trained weights file named **my_weights.pt**, you can easily run the model using the following command:

```bash
python run.py --weights my_weights.pt --count 100
```

If you want to decode text and visualize the learned 2D vector alignments of your character embeddings simultaneously, add the `--plot` flag:

```bash
python run.py --weights virtual_usb_drive.pt --plot
```

---

## Launching the Training

To train the model from scratch, you can pass your custom hyperparameters directly through the command-line interface:

```bash
python train.py --iter 50000 --block_size 5 --dim 10 --output_mode steps --save_to ma_cle.pt
```

Alternatively, you can run the pipeline with the default settings:

```bash
python train.py
```

To explore all available parameters, command flags, and details about the architecture, view the summary document string by running:

```bash
python train.py --help
```
