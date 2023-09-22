# Segmentation for Tools:

Manual to use notebook:

### 1. Install conda environment and dependencies:

- when conda is installed type in (anaconda) console: 
- conda -n create "environment name here" python=3.10 (or any other python version in fact, checked with 3.10)
- conda activate "environment name here"
- pip install -r requirements.txt

### 2. Run Notebook (is divided in guided steps). Keep in mind the structure in data is hard coded. For changes in folder name changes in code are necessary.

### 3. If new train images need to be added with labels:

- conda activate "environment name here"
- labelme -> labelme as command opens labelme
- draw polygon and save the name of the label as "wear" 
- move both image jpg and label json to train folder (make sure they have the same name excluding the ending!)