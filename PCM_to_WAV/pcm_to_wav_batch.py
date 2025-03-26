from PCM_to_WAV.pcm_to_wav import pcm_to_wav as ptw
import glob, os, sys


if __name__ == '__main__':
    source_dir = '/DATA/PCM'
    target_dir = '/DATA/WAV'

    #parent directory
    par_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    if not os.path.isdir(par_dir+source_dir):
        print(f'Missing source directory: {par_dir+source_dir}')
        sys.exit(1)

    if not os.path.isdir(par_dir+target_dir):
        os.mkdir(par_dir+target_dir)
        print(f'Missing target directory. Creating: {target_dir}')

    # change current directory to source directory
    os.chdir(par_dir+source_dir)
    idx = 0

    #Iterate through current directory for pcm files and conert them to wav each
    for file in glob.glob("*.pcm"):
        ptw.pcm_to_wav(file_path=file,source_dir=par_dir+source_dir,target_dir=par_dir+target_dir)
        print(f"{file} was converted. {idx+1}")
        idx += 1

