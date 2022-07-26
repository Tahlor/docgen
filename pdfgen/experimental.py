def create_new_text_reportlab():
    """ Read original pdf, get dimensions, create copy, fill in with text, then merge

    Returns:

    """
    from reportlab.pdfgen import canvas
    from PyPDF2 import PdfFileWriter, PdfFileReader
    c = canvas.Canvas('watermark.pdf')
    c.drawImage('ttest.png', 15, 720)
    c.drawString(15, 720, "Hello World")
    c.save()

def create_watermark():
    from reportlab.pdfgen import canvas
    from PyPDF2 import PdfFileWriter, PdfFileReader
    # Create the watermark from an image
    c = canvas.Canvas('watermark.pdf')
    c.drawImage('ttest.png', 15, 720)
    c.drawString(15, 720, "Hello World")
    c.save()

def merge(source, watermark, output, page=1):
    watermark = PdfFileReader(open(watermark, "rb"))
    output_file = PdfFileWriter()
    input_file = PdfFileReader(open(source, "rb"))
    input_page = input_file.getPage(page)
    input_page.mergePage(watermark.getPage(0))
    # add page from input file to output document
    output_file.addPage(input_page)

    # finally, write "output" to document-output.pdf
    with open(output, "wb") as outputStream:
        output_file.write(outputStream)
