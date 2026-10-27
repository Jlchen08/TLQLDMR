# TLQLDMR

## Git LFS 与可复现性补充包

`TLQLDMR_R1/Supplementary_Reproducibility_S1.zip` 由 Git LFS 管理。首次克隆前请安装 Git LFS；克隆后执行：

```bash
git lfs install
git lfs pull
```

该压缩包的 SHA-256 为 `262efbf7da9740206f3a840e4adc00240d41609c55689873afe831a2814d2b59`。解压副本、逐模型预测目录、选择缓存和 pytest 沙箱均可由已提交的脚本或压缩包恢复，因此不进入普通 Git 历史。仓库直接保留正式 v2 的分析汇总、渲染结果、验证报告及关键锁文件，便于无需展开完整补充包的审阅。

## LaTeX 编译

使用 TeX Live（或含 `latexmk`、Perl 的 MiKTeX），并确保终端和 VS Code 的运行环境都能找到 `latexmk`、`pdflatex`、`bibtex`。本项目统一使用 pdfLaTeX，参考文献由 latexmk 按需调用 BibTeX。

在仓库根目录编译全部文档：

```bash
latexmk
```

根目录的 `latexmkrc` 会依次编译下表中的 9 个入口，将 PDF、SyncTeX 和辅助文件写入各自目录的 `build/`。也可以进入任一文档目录运行 `latexmk`，编译该目录的全部入口。

| 目录 | 编译入口 |
| --- | --- |
| `TLQLDMR_manuscript` | `TLQLDMR.tex`、`TLQLDMR_energy.tex`、`framework.tex`、`Response_to_Technical_Check_Results.tex` |
| `TLQLDMR_R1` | `TLQLDMR_energy.tex`、`framework.tex`、`Response_to_Technical_Check_Results.tex`、`experiments-preview.tex` |
| `TLQLDMR_response_1st` | `main.tex` |

### 手动编译单篇文档

在仓库根目录：

```bash
latexmk -pdf TLQLDMR_R1/TLQLDMR_energy.tex
latexmk -pdf TLQLDMR_response_1st/main.tex
```

也可以从任意目录使用完整文件路径：

```bash
latexmk -cd -pdf -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error -outdir=build /path/to/TLQLDMR/TLQLDMR_R1/TLQLDMR_energy.tex
```

`-cd` 会先切换到主文件所在目录，因此图片、模板和参考文献的相对路径不受启动目录影响。含空格的文件路径需用引号包围。

若需要直接调用引擎，在文档目录内执行（以 R1 论文为例）：

```bash
cd TLQLDMR_R1
mkdir -p build
pdflatex -synctex=1 -interaction=nonstopmode -halt-on-error -output-directory=build TLQLDMR_energy.tex
bibtex build/TLQLDMR_energy
pdflatex -synctex=1 -interaction=nonstopmode -halt-on-error -output-directory=build TLQLDMR_energy.tex
pdflatex -synctex=1 -interaction=nonstopmode -halt-on-error -output-directory=build TLQLDMR_energy.tex
```

若日志仍提示交叉引用改变，再运行一次 `pdflatex`。不含参考文献的框图和技术检查回复信无需执行 BibTeX。`latexmk` 会自动处理这些区别及所需轮数。

### LaTeX Workshop

安装 VS Code 扩展 `James-Yu.latex-workshop`。仓库根目录和三个文档目录均提供 `.vscode/settings.json`，支持打开整个仓库或单独打开一个文档目录。

打开需要编译的 TeX 文件，先点击正文使其获得焦点，再执行 **LaTeX Workshop: Build LaTeX project**（通常为 `Ctrl+Alt+B`），默认配方为 `latexmk (pdfLaTeX)`。随后执行 **View LaTeX PDF file** 查看对应 `build/` 下的 PDF。

默认配方通过 `bash -lc` 启动 `latexmk`，加载 Linux/Remote SSH 的登录环境，避免扩展进程缓存的旧 PATH 找不到 TeX。参数以 `"$@"` 原样传递，支持路径中的空格。Windows 或已配置好系统 PATH 的环境可以选择备用配方 `latexmk (system PATH)`。

若输出出现 `Can't find LaTeX root file`，同时活动文档是 `output:...`、语言是 `latex_workshop_log`，说明焦点还在输出面板。此时点击 `.tex` 正文，再点编译按钮；无需修改论文的根文件路径。

### 从 PDF 跳回 TeX 源码

仓库根目录和三个文档目录均已设置：使用 LaTeX Workshop 内置 PDF 预览，**双击 PDF 正文即可跳到对应 TeX 文件及行号**。保存源码会自动编译；编译参数包含 `-synctex=1`，定位所需的 `.synctex.gz` 与 PDF 一起保存在 `build/` 中，扩展的辅助文件清理列表会保留该定位文件。

1. 打开 `.tex`，执行 **LaTeX Workshop: Build LaTeX project**（通常为 `Ctrl+Alt+B`）。
2. 执行 **LaTeX Workshop: View LaTeX PDF file**（通常为 `Ctrl+Alt+V`），打开当前生成的 PDF。
3. 在 PDF 的文字、公式或图表附近双击，编辑器会打开对应源码并定位到相关行。对于 `\input` 引入的章节，定位目标是实际的章节文件。

从源码定位到 PDF 可使用 `Ctrl+Alt+J`。这些操作使用 [LaTeX Workshop 的 SyncTeX 功能](https://github.com/James-Yu/LaTeX-Workshop/wiki/View#synctex)。

SyncTeX 按排版位置定位：长表格的页眉、页脚等内容可能跳到对应表格环境的起止行；导入图片会定位到相关的 `\includegraphics` 命令。

请使用当前 `build/` 下的 PDF，并保留同名 `.synctex.gz`；单独复制 PDF 或打开历史编译副本无法保证源码定位。若 PDF 已在其他预览器中打开，右键标签页选择 **Reopen Editor With… → LaTeX Workshop Internal PDF Viewer**。如果修改设置后旧预览标签未更新，关闭后重新执行上述预览命令。

两份 `experiments.tex` 是章节片段，不是完整的独立文档：

- `TLQLDMR_manuscript/experiments.tex` 的 root 是同目录的 `TLQLDMR.tex`。
- `TLQLDMR_R1/experiments.tex` 的 root 是新增的 `experiments-preview.tex`。R1 正式论文保留自己的内联实验章节；修改这个片段只影响预览。

LaTeX Workshop 会读取文件顶部的 `% !TEX root`，编译正确的入口。命令行应直接编译相应入口，不要对章节片段单独运行 `pdflatex`。

R1 实验预览通过 `xr` 读取 `build/TLQLDMR_energy.aux` 中的方法章节编号。首次预览、清理输出后或修改正式论文编号后，先编译 `TLQLDMR_energy.tex`，再编译 `experiments-preview.tex`；根目录及 R1 目录的 `latexmk` 默认顺序已处理此依赖。

### 资源与环境

- 两份 `framework.tex` 是独立框图入口；论文与框图共用各自目录的 `framework.tikz`。论文首次构建不依赖预先生成的 `build/framework.pdf`。
- 两份论文目录均包含配套的 `cas-sc.cls`、`cas-common.sty`，避免依赖机器上另一个版本的 CAS 模板。
- 精简安装至少需要 `latexmk`、LaTeX 基础/推荐宏包、`els-cas-templates` 的依赖、`standalone`、`pgf`、`ieeetran`、`algorithm2e`、`ifoddpage`、`relsize`、`natbib`、`caption`、`multirow`、`makecell`、`placeins`、`threeparttable`、`jknapltx`、`rsfs`、`stix`、`times` 等；完整 TeX Live 通常已包含。`cas-sc` 检测不到可选 Charis 字体时可使用 STIX。
- 当前 Linux 环境的用户级 TeX Live 安装位于 `~/.local/texlive/2026`，登录配置会加载 `~/.local/texlive/env.sh`。现有终端可执行 `. "$HOME/.local/texlive/env.sh"` 立即使用。
- Remote SSH 的扩展进程可能始终使用不含用户 TeX 的 PATH，单纯重连窗口不一定解决 `spawn latexmk ENOENT`。默认登录 Shell 配方已处理这种情况；若选择备用 `latexmk (system PATH)`，则必须确保扩展进程本身的 PATH 正确。只在终端 `export PATH` 不会更新已运行的扩展进程。
- 仓库保留了部分历史编译产物。当前配置使用各文档目录的 `build/`；不要把旧位置的 PDF 当作本次构建结果。如旧缓存导致异常，可在相应目录运行 `latexmk -gg` 清理后重建。

构建选项参见 [latexmk 手册](https://www.ctan.net/support/latexmk/latexmk.txt)，编辑器配置参见 [LaTeX Workshop 编译文档](https://github.com/James-Yu/LaTeX-Workshop/wiki/Compile)。

### 已验证的构建方式

在 TeX Live 2026、latexmk 4.88 下，9 个入口均已生成 PDF，两个章节片段均由对应入口覆盖。验证包括：

- 仓库根目录运行 `latexmk`，以及无旧缓存、路径含空格的副本全量构建。
- LaTeX Workshop 10.20.0 的实际 Recipe/Plan 参数展开后执行构建，覆盖仓库根工作区和三个独立子目录工作区，检查 PDF 与 SyncTeX 输出位置。
- 按按钮首次失败时远程 Extension Host 的实际 PATH（不含 TeX）复测登录 Shell 配方，在含空格路径的副本中强制重编译全部 9 个入口，均成功。
- 论文和回复信直接运行 `pdflatex → bibtex → pdflatex` 多轮构建。

最终日志无缺失文件、未定义引用或未闭合分组；仍有部分排版提示和回复信的未设置作者/标题宏包兼容性提示。编辑器配方执行验证与 GUI 点击操作应分别确认。
