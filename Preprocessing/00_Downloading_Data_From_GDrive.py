import gdown

### The URL to the project folder on my personal Google Drive. Contains mainly the Original Data Files only.
url = 'https://drive.google.com/drive/folders/1wpIGXkWbPdrzFpv8k_nDR-g3aVOjyytP'

gdown.download_folder(url, quiet=True)

### Now I'll go place this folder somewhere nice
