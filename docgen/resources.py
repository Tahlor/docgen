from pathlib import Path
import site

DOCGEN = Path(site.getsitepackages()[0]) / "docgen"

def download_handwriting(force=False):
    from download_resources.download import download_s3_folder, directory_is_empty
    if force or not DOCGEN.exists() or directory_is_empty(DOCGEN):

        s3_hwr = "s3://datascience-computervision-l3apps/HWR/synthetic-data/python-package-resources/generated-handwriting/single-word-datasets/iam-cvl-32px-top10k-words/style_600_IAM_IAM_samples.npy"
        download_s3_folder(s3_hwr, DOCGEN)
    return DOCGEN

def download_handwriting_zip(force=False):
    from download_resources.download import download_s3_folder, directory_is_empty
    if force or not DOCGEN.exists() or directory_is_empty(DOCGEN):
        s3_hwr = "s3://datascience-computervision-l3apps/HWR/synthetic-data/python-package-resources/generated-handwriting/single-word-datasets/iam-cvl-32px-top10k-words/synth_hw.zip"
        download_s3_folder(s3_hwr, DOCGEN)
    return DOCGEN


if __name__=='__main__':
    pass