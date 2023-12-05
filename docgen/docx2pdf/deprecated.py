import sys
import subprocess
import re
#import docx2pdf

LINUX_PATH = "/usr/bin/libreoffice7.3"
MACOS_PATH = "/Applications/LibreOffice.app/Contents/MacOS/soffice"

def convert_docx_to_pdf(input_path, output_path=None):
    """

    Args:
        input_file:
        output_file:

    Returns:

    """
    if sys.platform in ["darwin", "win32"]:
        return docx2pdf.convert(input_path=input_path, output_path=output_path, keep_active=False)
    else:
        return docx2pdf_oo(input_path=input_path, output_path=output_path)

def docx2pdf_oo(input_path, output_path, timeout=None):
    """

    Args:
        input_path:
        output_path:
        timeout:

    Returns:

    """
    args = [libreoffice_exec(), '--headless', '--convert-to', 'pdf', '--outdir', output_path, input_path]

    process = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    filename = re.search('-> (.*?) using filter', process.stdout.decode())

    return filename.group(1)


def libreoffice_exec():
    # TODO: Provide support for more platforms
    if sys.platform == 'linux':
        return LINUX_PATH
    if sys.platform == 'darwin':
        return MACOS_PATH
    return 'libreoffice'
