import sys
import json
import subprocess
from pathlib import Path
from tqdm.auto import tqdm
import re


""" A gently modified version of docx2pdf to work with Open Office on Linux

"""

# try:
#     # 3.8+
#     from importlib.metadata import version
# except ImportError:
#     from importlib_metadata import version
#
# __version__ = version(__package__)

LINUX_PATH = "/usr/bin/libreoffice"
MACOS_PATH = "/Applications/LibreOffice.app/Contents/MacOS/soffice"


def windows(paths, keep_active, word=None):
    import win32com.client

    if word is None:
        word = win32com.client.Dispatch("Word.Application")
    else:
        keep_active = True

    wdFormatPDF = 17

    def convert():
        doc = word.Documents.Open(str(docx_filepath))
        doc.SaveAs(str(pdf_filepath), FileFormat=wdFormatPDF)
        doc.Close(0)

    if paths["batch"]:
        for docx_filepath in tqdm(sorted(Path(paths["input"]).glob("*.docx"))):
            pdf_filepath = Path(paths["output"]) / (str(docx_filepath.stem) + ".pdf")
            convert()
    else:
        pbar = tqdm(total=1)
        docx_filepath = Path(paths["input"]).resolve()
        pdf_filepath = Path(paths["output"]).resolve()
        convert()
        pbar.update(1)

    if not keep_active:
        word.Quit()


def macos(paths, keep_active):
    script = (Path(__file__).parent / "convert.jxa").resolve()
    cmd = [
        "/usr/bin/osascript",
        "-l",
        "JavaScript",
        str(script),
        str(paths["input"]),
        str(paths["output"]),
        str(keep_active).lower(),
    ]

    def run(cmd):
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        while True:
            line = process.stderr.readline().rstrip()
            if not line:
                break
            yield line.decode("utf-8")

    total = len(list(Path(paths["input"]).glob("*.docx"))) if paths["batch"] else 1
    pbar = tqdm(total=total)
    for line in run(cmd):
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg["result"] == "success":
            pbar.update(1)
        elif msg["result"] == "error":
            print(msg)
            sys.exit(1)


def resolve_paths(input_path, output_path):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve() if output_path else None
    output = {}
    if input_path.is_dir():
        output["batch"] = True
        output["input"] = str(input_path)
        if output_path:
            assert output_path.is_dir()
        else:
            output_path = str(input_path)
        output["output"] = output_path
    else:
        output["batch"] = False
        assert str(input_path).endswith(".docx")
        output["input"] = str(input_path)
        if output_path and output_path.is_dir():
            output_path = str(output_path / (str(input_path.stem) + ".pdf"))
        elif output_path:
            assert str(output_path).endswith(".pdf")
        else:
            output_path = str(input_path.parent / (str(input_path.stem) + ".pdf"))
        output["output"] = output_path
    return output


def docx2pdf_oo(paths, keep_active, office_path=None):
    """

    Args:
        input_path:
        output_path:
        timeout:

    Returns:

    """
    def convert():
        _pdf_filepath = Path(pdf_filepath).parent if Path(pdf_filepath).suffix == ".pdf" else pdf_filepath
        args = [libreoffice_exec() if office_path is None else office_path, '--headless', '--convert-to', 'pdf', '--outdir', _pdf_filepath, docx_filepath]
        process = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        #filename = re.search('-> (.*?) using filter', process.stdout.decode())
        #filename.group(1)

    if paths["batch"]:
        for docx_filepath in tqdm(sorted(Path(paths["input"]).glob("*.docx"))):
            pdf_filepath = Path(paths["output"]) / (str(docx_filepath.stem) + ".pdf")
            convert()
    else:
        pbar = tqdm(total=1)
        docx_filepath = Path(paths["input"]).resolve()
        pdf_filepath = Path(paths["output"]).resolve()
        convert()
        pbar.update(1)


def libreoffice_exec():
    if sys.platform == 'linux':
        return LINUX_PATH
    if sys.platform == 'darwin':
        return MACOS_PATH
    return 'libreoffice'


def convert(input_path, output_path=None, keep_active=False, office_path=None, word=None):
    paths = resolve_paths(input_path, output_path)
    if not office_path is None:
        return docx2pdf_oo(paths, keep_active, office_path=office_path)
    if sys.platform == "darwin":
        return macos(paths, keep_active)
    elif sys.platform == "win32":
        return windows(paths, keep_active, word=word)
    elif sys.platform == "linux":
        return docx2pdf_oo(paths, keep_active)
    else:
        raise NotImplementedError(
            "docx2pdf is not implemented for linux as it requires Microsoft Word to be installed"
        )


def cli():

    import textwrap
    import argparse

    if "--version" in sys.argv:
        print(__version__)
        sys.exit(0)

    description = textwrap.dedent(
        """
    Example Usage:

    Convert single docx file in-place from myfile.docx to myfile.pdf:
        docx2pdf myfile.docx

    Batch convert docx folder in-place. Output PDFs will go in the same folder:
        docx2pdf myfolder/

    Convert single docx file with explicit output filepath:
        docx2pdf input.docx output.docx

    Convert single docx file and output to a different explicit folder:
        docx2pdf input.docx output_dir/

    Batch convert docx folder. Output PDFs will go to a different explicit folder:
        docx2pdf input_dir/ output_dir/
    """
    )

    formatter_class = lambda prog: argparse.RawDescriptionHelpFormatter(
        prog, max_help_position=32
    )
    parser = argparse.ArgumentParser(
        description=description, formatter_class=formatter_class
    )
    parser.add_argument(
        "input",
        help="input file or folder. batch converts entire folder or convert single file",
    )
    parser.add_argument("output", nargs="?", help="output file or folder")
    parser.add_argument(
        "--keep-active",
        action="store_true",
        default=False,
        help="prevent closing word after conversion",
    )
    parser.add_argument(
        "--version", action="store_true", default=False, help="display version and exit"
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    else:
        args = parser.parse_args()

    convert(args.input, args.output, args.keep_active)
