# Resolve all inputs relative to the document, even from the repository root.
$do_cd = 1;
$pdf_mode = 1;
$out_dir = 'build';
$aux_dir = 'build';
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error %O %S';

# A plain latexmk invocation builds every document, including chapter previews.
@default_files = (
    'TLQLDMR_manuscript/framework.tex',
    'TLQLDMR_manuscript/Response_to_Technical_Check_Results.tex',
    'TLQLDMR_manuscript/TLQLDMR.tex',
    'TLQLDMR_manuscript/TLQLDMR_energy.tex',
    'TLQLDMR_R1/framework.tex',
    'TLQLDMR_R1/Response_to_Technical_Check_Results.tex',
    'TLQLDMR_R1/TLQLDMR_energy.tex',
    'TLQLDMR_R1/experiments-preview.tex',
    'TLQLDMR_response_1st/main.tex',
);
