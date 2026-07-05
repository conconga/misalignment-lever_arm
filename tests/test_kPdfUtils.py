#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
import sys
print( "**************************************" )
print(f"** __name__    = {__name__}")
print(f"** __package__ = {__package__}")
print(f"** sys.path[0] = {sys.path[0]}")

import pytest
from kPdfUtils import kPdfUtils
from unittest.mock import MagicMock, patch

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
class TestClass_kPdfUtils:

    def test_export_figures_2_pdf(self):
        pdf_cm = MagicMock()
        pdf_cm.__enter__.return_value = pdf_cm

        with patch("kPdfUtils.pdf", return_value=pdf_cm) as pdf_mock, \
            patch("kPdfUtils.plt.get_fignums", return_value=[101, 202]) as get_fignums_mock:

                kPdfUtils.export_figures_2_pdf("out.pdf")

                pdf_cm.savefig.assert_any_call(101)
                pdf_cm.savefig.assert_any_call(202)
                assert pdf_cm.savefig.call_count == 2

    def test_fix_filename_extension(self):

        # no change:
        assert kPdfUtils.fix_filename_extension("a.pdf") == "a.pdf"

        # no change:
        assert kPdfUtils.fix_filename_extension("a.pDf") == "a.pDf"

        # stripping:
        assert kPdfUtils.fix_filename_extension("  a.pDf   ") == "a.pDf"

        # incomplete extension:
        assert kPdfUtils.fix_filename_extension("a.pd") == "a.pd.pdf"
        assert kPdfUtils.fix_filename_extension("a.") == "a..pdf"
        assert kPdfUtils.fix_filename_extension("a") == "a.pdf"


#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
