#//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages as pdf
#//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#


class kPdfUtils:

    @classmethod
    def export_figures_2_pdf(cls, filename):
        with pdf(filename) as f:
            for fig in plt.get_fignums():
                f.savefig(fig)

    @classmethod
    def fix_filename_extension(cls, filename):
        re_pdf = re.compile(r'\.pdf$', re.IGNORECASE)
        if not re_pdf.search(filename.strip()):
            return filename.strip() + ".pdf"
        else:
            return filename.strip()

#//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#
#//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#
