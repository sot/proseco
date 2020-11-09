# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.io import ascii
from ..diff import catalog_diff, get_catalog_lines


CAT1 = ascii.read("""
slot idx  id type  sz p_acq  mag  maxmag   yang     zang   dim res halfw
---- --- --- ---- --- ----- ----- ------ -------- -------- --- --- -----
   0   1   2  FID 8x8 0.000  7.00   8.00  -773.20 -1742.03   1   1    25
   1   2   4  FID 8x8 0.000  7.00   8.00  2140.23   166.63   1   1    25
   2   3   5  FID 8x8 0.000  7.00   8.00 -1826.28   160.17   1   1    25
   3   4 101  BOT 6x6 0.935  9.50  11.00     0.00  1500.00  20   1   160
   4   5 105  BOT 6x6 0.981  7.50   9.00   750.00  -750.00  20   1   160
   5   6 106  BOT 6x6 0.982  7.00   8.50  -750.00   750.00  20   1   160
   6   7 107  BOT 6x6 0.982  6.50   8.00  -750.00  -750.00  20   1   160
   7   8 103  GUI 6x6 0.000  8.50  10.00     0.00 -1500.00   1   1    25
   7   9 104  ACQ 6x6 0.980  8.00   9.50   750.00   750.00  20   1   160
   0  10 102  ACQ 6x6 0.966  9.00  10.50 -1500.00     0.00  20   1   160
   1  11 108  ACQ 6x6 0.551 10.50  11.20   300.00   300.00  20   1    60
""")

CAT2 = ascii.read("""
slot idx  id type  sz p_acq  mag  maxmag   yang     zang   dim res halfw
---- --- --- ---- --- ----- ----- ------ -------- -------- --- --- -----
   0   1   2  FID 8x8 0.000  7.00   8.00  -773.20 -1742.03   1   1    25
   1   2   4  FID 8x8 0.000  7.00   8.00  2140.23   166.63   1   1    25
   2   3   5  FID 8x8 0.000  7.00   8.00 -1826.28   160.17   1   1    25
   3   4 101  BOT 6x6 0.935  9.50  11.00     0.00  1500.00  20   1   160
   4   5 105  BOT 6x6 0.981  7.50   9.00   750.00  -750.00  20   1   160
   5   6 106  BOT 6x6 0.982  7.00   8.50  -750.00   750.00  20   1   160
   6   7 100  MON 8x8 0.982  6.50   8.00  -750.00  -750.00  20   1   160
   7   8 103  GUI 6x6 0.000  8.50  10.00     0.00 -1500.00   1   1    25
   7   9 104  ACQ 6x6 0.980  8.00   9.50   750.00   750.00  20   1   160
   0  10 102  ACQ 6x6 0.966  9.00  10.50 -1500.00     0.00  20   1   160
   1  11 108  ACQ 6x6 0.551 10.50  11.20   300.00   300.00  20   1    60
""")


def test_diff_unified():
    diff = catalog_diff(CAT1, CAT2, style='unified')
    # Note: sorted by mag within each section
    assert diff.text.splitlines() == [
        '--- catalog 1',
        '',
        '+++ catalog 2',
        '',
        '@@ -8,12 +8,12 @@',
        '',
        '   8 103    7  GUI  8.5 6x6   1   1    25',
        '   5 105    4  GU*  7.5 6x6   1   1    25',
        '   6 106    5  GU*  7.0 6x6   1   1    25',
        '-  7 107    6  GU*  6.5 6x6   1   1    25',
        '+--- --- ---- ---- ---- --- --- --- -----',
        '+  7 100    6  MON  6.5 8x8  20   1    25',
        ' --- --- ---- ---- ---- --- --- --- -----',
        '   4 101    3  AC*  9.5 6x6  20   1   160',
        '  10 102    0  ACQ  9.0 6x6  20   1   160',
        '   9 104    7  ACQ  8.0 6x6  20   1   160',
        '   5 105    4  AC*  7.5 6x6  20   1   160',
        '   6 106    5  AC*  7.0 6x6  20   1   160',
        '-  7 107    6  AC*  6.5 6x6  20   1   160',
        '  11 108    1  ACQ 10.5 6x6  20   1    60',
    ]


def test_diff_context():
    diff = catalog_diff(CAT1, CAT2, style='context')
    # Note: sorted by mag within each section
    assert diff.text.splitlines() == [
        '*** catalog 1',
        '',
        '--- catalog 2',
        '',
        '***************',
        '',
        '*** 8,19 ****',
        '',
        '    8 103    7  GUI  8.5 6x6   1   1    25',
        '    5 105    4  GU*  7.5 6x6   1   1    25',
        '    6 106    5  GU*  7.0 6x6   1   1    25',
        '!   7 107    6  GU*  6.5 6x6   1   1    25',
        '  --- --- ---- ---- ---- --- --- --- -----',
        '    4 101    3  AC*  9.5 6x6  20   1   160',
        '   10 102    0  ACQ  9.0 6x6  20   1   160',
        '    9 104    7  ACQ  8.0 6x6  20   1   160',
        '    5 105    4  AC*  7.5 6x6  20   1   160',
        '    6 106    5  AC*  7.0 6x6  20   1   160',
        '-   7 107    6  AC*  6.5 6x6  20   1   160',
        '   11 108    1  ACQ 10.5 6x6  20   1    60',
        '--- 8,19 ----',
        '',
        '    8 103    7  GUI  8.5 6x6   1   1    25',
        '    5 105    4  GU*  7.5 6x6   1   1    25',
        '    6 106    5  GU*  7.0 6x6   1   1    25',
        '! --- --- ---- ---- ---- --- --- --- -----',
        '!   7 100    6  MON  6.5 8x8  20   1    25',
        '  --- --- ---- ---- ---- --- --- --- -----',
        '    4 101    3  AC*  9.5 6x6  20   1   160',
        '   10 102    0  ACQ  9.0 6x6  20   1   160',
        '    9 104    7  ACQ  8.0 6x6  20   1   160',
        '    5 105    4  AC*  7.5 6x6  20   1   160',
        '    6 106    5  AC*  7.0 6x6  20   1   160',
        '   11 108    1  ACQ 10.5 6x6  20   1    60',
    ]


def test_diff_section_lines():
    diff = catalog_diff(CAT1, CAT2, style='unified', section_lines=False)
    # No line separating the sections
    assert diff.text.splitlines() == [
        '--- catalog 1',
        '',
        '+++ catalog 2',
        '',
        '@@ -7,11 +7,10 @@',
        '',
        '   8 103    7  GUI  8.5 6x6   1   1    25',
        '   5 105    4  GU*  7.5 6x6   1   1    25',
        '   6 106    5  GU*  7.0 6x6   1   1    25',
        '-  7 107    6  GU*  6.5 6x6   1   1    25',
        '+  7 100    6  MON  6.5 8x8  20   1    25',
        '   4 101    3  AC*  9.5 6x6  20   1   160',
        '  10 102    0  ACQ  9.0 6x6  20   1   160',
        '   9 104    7  ACQ  8.0 6x6  20   1   160',
        '   5 105    4  AC*  7.5 6x6  20   1   160',
        '   6 106    5  AC*  7.0 6x6  20   1   160',
        '-  7 107    6  AC*  6.5 6x6  20   1   160',
        '  11 108    1  ACQ 10.5 6x6  20   1    60',
    ]


def test_diff_sort_name():
    diff = catalog_diff(CAT1, CAT2, style='unified', sort_name='mag')
    # Note: sorted by mag within each section
    assert diff.text.splitlines() == [
        '--- catalog 1',
        '',
        '+++ catalog 2',
        '',
        '@@ -4,13 +4,13 @@',
        '',
        '   2   4    1  FID  7.0 8x8   1   1    25',
        '   3   5    2  FID  7.0 8x8   1   1    25',
        ' --- --- ---- ---- ---- --- --- --- -----',
        '-  7 107    6  GU*  6.5 6x6   1   1    25',
        '   6 106    5  GU*  7.0 6x6   1   1    25',
        '   5 105    4  GU*  7.5 6x6   1   1    25',
        '   8 103    7  GUI  8.5 6x6   1   1    25',
        '   4 101    3  GU*  9.5 6x6   1   1    25',
        ' --- --- ---- ---- ---- --- --- --- -----',
        '-  7 107    6  AC*  6.5 6x6  20   1   160',
        '+  7 100    6  MON  6.5 8x8  20   1    25',
        '+--- --- ---- ---- ---- --- --- --- -----',
        '   6 106    5  AC*  7.0 6x6  20   1   160',
        '   5 105    4  AC*  7.5 6x6  20   1   160',
        '   9 104    7  ACQ  8.0 6x6  20   1   160'
    ]


def test_diff_get_catalog_lines():
    names = 'slot idx id type  sz p_acq  mag zang dim halfw'
    lines = get_catalog_lines(CAT2, names=names, sort_name='mag')
    assert lines == [
        'slot idx  id type  sz p_acq mag    zang   dim halfw',
        '---- --- --- ---- --- ----- ---- -------- --- -----',
        '   0   1   2  FID 8x8   0.0  7.0 -1742.03   1    25',
        '   1   2   4  FID 8x8   0.0  7.0   166.63   1    25',
        '   2   3   5  FID 8x8   0.0  7.0   160.17   1    25',
        '---- --- --- ---- --- ----- ---- -------- --- -----',
        '   5   6 106  GU* 6x6 0.982  7.0    750.0   1    25',
        '   4   5 105  GU* 6x6 0.981  7.5   -750.0   1    25',
        '   7   8 103  GUI 6x6   0.0  8.5  -1500.0   1    25',
        '   3   4 101  GU* 6x6 0.935  9.5   1500.0   1    25',
        '---- --- --- ---- --- ----- ---- -------- --- -----',
        '   6   7 100  MON 8x8 0.982  6.5   -750.0  20    25',
        '---- --- --- ---- --- ----- ---- -------- --- -----',
        '   5   6 106  AC* 6x6 0.982  7.0    750.0  20   160',
        '   4   5 105  AC* 6x6 0.981  7.5   -750.0  20   160',
        '   7   9 104  ACQ 6x6  0.98  8.0    750.0  20   160',
        '   0  10 102  ACQ 6x6 0.966  9.0      0.0  20   160',
        '   3   4 101  AC* 6x6 0.935  9.5   1500.0  20   160',
        '   1  11 108  ACQ 6x6 0.551 10.5    300.0  20    60'
    ]


def test_diff_html():
    """Test that it runs without exception, functional test separately in
    Jupyter notebook"""
    catalog_diff([CAT1, CAT2], [CAT2, CAT1], style='html')