from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfform
from reportlab.lib.colors import magenta, pink, blue, green
from pathlib import Path
import os
import sys

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

folder = Path(os.path.dirname(__file__))

def create_simple_form():
    c = canvas.Canvas('simple_form.pdf')

    c.setFont("Courier", 14)
    form = c.acroForm

    c.drawString(10, 650, 'First Name:')

    c.setFont("Courier", 8)
    c.drawString(110, 670, 'First Name:')

    form.textfield(name='fname', tooltip='First Name',
                   x=110, y=635, borderStyle='inset',
                   borderColor=None, fillColor=None,
                   width=300,
                   textColor=None, forceBorder=True)

    c.drawString(10, 600, 'Last Name:')
    form.textfield(name='lname', tooltip='Last Name',
                   x=110, y=585, borderStyle='inset',
                   borderColor=None, fillColor=None,
                   width=300,
                   textColor=None, forceBorder=True)

    c.drawString(10, 550, 'Address:')
    form.textfield(name='address', tooltip='Address',
                   x=110, y=535, borderStyle='inset',
                   width=400, forceBorder=True)

    c.drawString(10, 500, 'City:')
    form.textfield(name='city', tooltip='City',
                   x=110, y=485, borderStyle='inset',
                   forceBorder=True)

    c.drawString(250, 500, 'State:')
    form.textfield(name='state', tooltip='State',
                   x=350, y=485, borderStyle='inset',
                   forceBorder=True)

    c.drawString(10, 450, 'Zip Code:')
    form.textfield(name='zip_code', tooltip='Zip Code',
                   x=110, y=435, borderStyle='inset',
                   forceBorder=True)

    c.save()



if __name__ == '__main__':
    create_simple_form()