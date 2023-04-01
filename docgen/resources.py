from pathlib import Path
import site
from download_resources.download import s3_download

DOCGEN = Path(site.getsitepackages()[0]) / "docgen/resources"
DOCGEN.mkdir(exist_ok=True, parents=True)


if __name__=='__main__':
    #download_handwriting_zip()
    download_handwriting(force=True)

