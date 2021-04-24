import sys
import os
from subprocess import call

if len(sys.argv) != 3:
  raise Exception('Incorrect command!, command is "python download DATASET_NAME PATH_TO_SAVE"')

DATASET = sys.argv[1]
PATH = sys.argv[2]

print('--- process ' + DATASET + ' dataset ---')

DATASET_PATH = os.path.join(PATH, DATASET)
CURRENT_PATH = os.getcwd()

if not os.path.exists(DATASET_PATH):
	os.makedirs(DATASET_PATH)
os.chdir(DATASET_PATH)

# download files
print("Downloading files....")

if DATASET == 'cars':
	call('wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz', shell=True)
	call('tar -zxf cars_train.tgz', shell=True)
	call('wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz', shell=True)
	call('tar -zxf car_devkit.tgz', shell=True)

elif DATASET == 'cub':
    try:
        from google_drive_downloader import GoogleDriveDownloader as gdd
    except:
        call('pip install googledrivedownloader', shell=True)
        from google_drive_downloader import GoogleDriveDownloader as gdd

    gdd.download_file_from_google_drive(file_id='1hbzc_P1FuxMkcabkgn9ZKinBwW683j45', dest_path='./CUB_200_2011.tgz', unzip=False)
    call('tar -zxf CUB_200_2011.tgz', shell=True)

elif DATASET == 'places':
	call('wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar', shell=True)
	call('tar -xf places365standard_easyformat.tar', shell=True)

elif DATASET == 'plantae':
	call('wget http://vllab.ucmerced.edu/ym41608/projects/CrossDomainFewShot/filelists/plantae.tar.gz', shell=True)
	call('tar -xzf plantae.tar.gz', shell=True)

else:
	raise Exception('No such dataset!')

# process file
print("Processing files....")

os.chdir(CURRENT_PATH)
call(f'python write_' + DATASET + f'_filelist.py {DATASET_PATH}', shell=True)
