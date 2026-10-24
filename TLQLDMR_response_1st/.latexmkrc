# Resolve all inputs relative to the document, even from the repository root.
$do_cd = 1;
$pdf_mode = 1;
$out_dir = 'build';
$aux_dir = 'build';
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error %O %S';

# Compile document roots only; experiments.tex is an included chapter.
@default_files = ('main.tex');

# A full system /tmp (root filesystem) breaks pdflatex, so keep its temporary
# files on the same roomy disk as the sources.
my $tmpdir = '/mnt/mydisk/zhangxiaohan/.latextmp';
mkdir $tmpdir unless -d $tmpdir;
$ENV{'TMPDIR'} = $tmpdir;
$ENV{'TEMP'}   = $tmpdir;
$ENV{'TMP'}    = $tmpdir;
