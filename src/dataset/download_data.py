import kagglehub
import shutil
from pathlib import Path
from src import utilities

def download_raw():
    dataset = utilities.dataset_select()['dataset']
    source = utilities.dataset_select()['source']

    # Download the dataset to the custom path
    check_raw = bool(utilities.check_raw(dataset))
    path = kagglehub.dataset_download(source)
    print("Đường dẫn dataset được tải về:", path)
    path_modified = Path(path) / dataset # path is currently a string, need Path to convert it to true Path else str error will appear

    base_dir = Path(__file__).parent.parent.parent
    destination = base_dir / 'data' / 'raw'

    if check_raw is False:
        shutil.move(path_modified,destination)
        print(f'Đường dần dataset được chuyển tới: {destination}')
        print("Tải về thành công")
    else:
        print("Dataset đã tồn tại")

    # Delete the original download path
    shutil.rmtree(Path(path).parent.parent.parent.parent.parent)

if __name__=="__main__":
    download_raw()